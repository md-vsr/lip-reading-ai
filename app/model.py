from __future__ import annotations

import math
import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import psutil
import torch
import torch.nn.functional as F

from app.config import DEFAULT_CHECKPOINT, THIRD_PARTY_ROOT, validate_model_source


@dataclass(frozen=True)
class WordCertainty:
    word: str
    certainty: float
    token_count: int


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    device: str
    model_load_seconds: float
    inference_seconds: float
    video_seconds: float
    real_time_factor: float
    decoding_score_per_token: float | None
    word_certainties: tuple[WordCertainty, ...]
    average_process_cpu_percent: float
    memory_rss_mb: float
    encoder_seconds: float = 0.0
    decoder_seconds: float = 0.0
    certainty_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def group_word_certainties(
    token_ids: list[int], token_log_probabilities: list[float], token_list: list[str]
) -> tuple[WordCertainty, ...]:
    """Group SentencePiece decoder probabilities into uncalibrated word estimates."""
    if len(token_ids) != len(token_log_probabilities):
        raise ValueError("Token IDs and token probabilities must have equal lengths.")

    words: list[WordCertainty] = []
    current_pieces: list[str] = []
    current_log_probabilities: list[float] = []

    def finish_word() -> None:
        if not current_pieces:
            return
        word = "".join(current_pieces).replace("▁", "").strip()
        if word:
            mean_log_probability = sum(current_log_probabilities) / len(
                current_log_probabilities
            )
            words.append(
                WordCertainty(
                    word=word,
                    certainty=min(1.0, max(0.0, math.exp(mean_log_probability))),
                    token_count=len(current_log_probabilities),
                )
            )
        current_pieces.clear()
        current_log_probabilities.clear()

    for token_id, log_probability in zip(token_ids, token_log_probabilities):
        piece = token_list[token_id]
        if piece in {"<eos>", "<blank>"}:
            finish_word()
            continue
        if piece.startswith("▁") and current_pieces:
            finish_word()
        current_pieces.append(piece)
        current_log_probabilities.append(log_probability)
    finish_word()
    return tuple(words)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot access Apple Metal.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, mps, cuda.")
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class MpsSpatialMaxPool(torch.nn.Module):
    """Apply the model's spatial-only 3D pool through MPS-supported 2D pooling."""

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("Expected video features shaped [B, C, T, H, W].")
        batch, channels, frames, height, width = video.shape
        # The temporal kernel is one: every B/C/T plane is independent. Avoid
        # permuting the full activation tensor before and after spatial pooling.
        spatial_frames = video.reshape(batch * channels * frames, 1, height, width)
        pooled = F.max_pool2d(
            spatial_frames,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        pooled_height, pooled_width = pooled.shape[-2:]
        return pooled.reshape(batch, channels, frames, pooled_height, pooled_width)


def _replace_mps_unsupported_pool(model: torch.nn.Module) -> None:
    pool = model.frontend.frontend3D[-1]
    if not isinstance(pool, torch.nn.MaxPool3d):
        raise RuntimeError(
            "The pinned Auto-AVSR pooling layer changed; refusing an unsafe MPS "
            "substitution."
        )
    expected = ((1, 3, 3), (1, 2, 2), (0, 1, 1))
    actual = (pool.kernel_size, pool.stride, pool.padding)
    if actual != expected:
        raise RuntimeError(
            "The pinned Auto-AVSR spatial pooling layer changed; refusing an unsafe "
            "MPS substitution."
        )
    model.frontend.frontend3D[-1] = MpsSpatialMaxPool()


class AutoAVSRRecognizer:
    """Thin inference wrapper around the pinned official Auto-AVSR model."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
        beam_size: int = 10,
        *,
        ctc_weight: float = 0.1,
        decoder_cache: bool = True,
        word_certainty: bool = True,
    ) -> None:
        validate_model_source()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run `python scripts/download_assets.py`."
            )
        if beam_size < 1:
            raise ValueError("Beam size must be at least 1.")
        if not math.isfinite(ctc_weight) or not 0 <= ctc_weight <= 1:
            raise ValueError("CTC weight must be between zero and one.")

        source = str(THIRD_PARTY_ROOT)
        if source not in sys.path:
            sys.path.insert(0, source)
        from lightning import ModelModule, get_beam_search_decoder

        self.encoder_device = choose_device(device)
        # Auto-AVSR's legacy ESPnet CTC prefix scorer creates CPU index tensors.
        # Keep decoding on CPU for MPS while accelerating the expensive visual
        # frontend and Conformer encoder on Metal.
        self.decoder_device = (
            torch.device("cpu")
            if self.encoder_device.type == "mps"
            else self.encoder_device
        )
        self.beam_size = beam_size
        started = time.perf_counter()
        args = Namespace(modality="video", ctc_weight=ctc_weight)
        module = ModelModule(args)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        module.model.load_state_dict(state_dict, strict=True)
        if self.encoder_device.type == "mps":
            _replace_mps_unsupported_pool(module.model)
        self.model = module.model.eval()
        self.model.frontend.to(self.encoder_device)
        self.model.proj_encoder.to(self.encoder_device)
        self.model.encoder.to(self.encoder_device)
        self.model.decoder.to(self.decoder_device)
        self.model.ctc.to(self.decoder_device)
        self.text_transform = module.text_transform
        self.token_list = module.token_list
        self.beam_search = get_beam_search_decoder(
            self.model,
            module.token_list,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
        )
        self.word_certainty = word_certainty
        self.cached_decoder = None
        if decoder_cache and ctc_weight < 1:
            from app.decoding import IncrementalDecoderScorer

            self.cached_decoder = IncrementalDecoderScorer(self.model.decoder).eval()
            self.beam_search.scorers["decoder"] = self.cached_decoder
            self.beam_search.full_scorers["decoder"] = self.cached_decoder
            self.beam_search.nn_dict["decoder"] = self.cached_decoder
        _synchronize(self.encoder_device)
        self.model_load_seconds = time.perf_counter() - started

    def transcribe(self, video: torch.Tensor, fps: float = 25.0) -> RecognitionResult:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Frame rate must be positive and finite.")
        if video.ndim != 4 or video.shape[1:] != (1, 88, 88):
            raise ValueError(
                f"Expected preprocessed video [T, 1, 88, 88], received {tuple(video.shape)}."
            )
        if len(video) < 3:
            raise ValueError("At least three preprocessed frames are required.")

        process = psutil.Process()
        cpu_before = process.cpu_times()
        started = time.perf_counter()
        sample = video.unsqueeze(0).to(self.encoder_device)
        with torch.inference_mode():
            encoded = self.model.frontend(sample)
            encoded = self.model.proj_encoder(encoded)
            encoded, _ = self.model.encoder(encoded, None)
            encoded = encoded.squeeze(0).to(self.decoder_device)
            _synchronize(self.encoder_device)
            encoder_finished = time.perf_counter()
            try:
                hypotheses = self.beam_search(encoded)
                _synchronize(self.decoder_device)
                decoder_finished = time.perf_counter()
                if not hypotheses:
                    raise RuntimeError(
                        "The decoder returned no transcription hypotheses."
                    )
                best_hypothesis = hypotheses[0]
                if not self.word_certainty:
                    word_certainties = tuple()
                elif self.cached_decoder is not None:
                    ids, probabilities = self.cached_decoder.token_scores(
                        best_hypothesis, self.model.eos
                    )
                    word_certainties = group_word_certainties(
                        ids.cpu().tolist(),
                        probabilities.cpu().tolist(),
                        self.token_list,
                    )
                else:
                    word_certainties = self._estimate_word_certainties(
                        best_hypothesis.yseq, encoded
                    )
            finally:
                if self.cached_decoder is not None:
                    self.cached_decoder.clear()
        _synchronize(self.encoder_device)
        inference_finished = time.perf_counter()
        inference_seconds = inference_finished - started
        cpu_after = process.cpu_times()
        cpu_seconds = (cpu_after.user + cpu_after.system) - (
            cpu_before.user + cpu_before.system
        )

        best = best_hypothesis.asdict()
        token_ids = torch.tensor([int(token) for token in best["yseq"][1:]])
        text = self.text_transform.post_process(token_ids).replace("<eos>", "").strip()
        score = best.get("score")
        if isinstance(score, torch.Tensor):
            score = float(score.detach().cpu())
        token_count = max(1, len(token_ids) - 1)
        score_per_token = float(score) / token_count if score is not None else None
        video_seconds = len(video) / fps

        return RecognitionResult(
            text=text,
            device=(
                str(self.encoder_device)
                if self.decoder_device == self.encoder_device
                else f"{self.encoder_device} encoder + {self.decoder_device} decoder"
            ),
            model_load_seconds=self.model_load_seconds,
            inference_seconds=inference_seconds,
            video_seconds=video_seconds,
            real_time_factor=inference_seconds / video_seconds,
            decoding_score_per_token=score_per_token,
            word_certainties=word_certainties,
            average_process_cpu_percent=100 * cpu_seconds / inference_seconds,
            memory_rss_mb=process.memory_info().rss / (1024 * 1024),
            encoder_seconds=encoder_finished - started,
            decoder_seconds=decoder_finished - encoder_finished,
            certainty_seconds=inference_finished - decoder_finished,
        )

    def _estimate_word_certainties(
        self, hypothesis_tokens: torch.Tensor, encoded: torch.Tensor
    ) -> tuple[WordCertainty, ...]:
        """Score each decoded word with teacher-forced decoder probabilities.

        The estimate is the geometric mean of the probabilities of the
        SentencePiece tokens forming a word. It is useful for relative
        uncertainty display but is not calibrated as correctness probability.
        """
        yseq = hypothesis_tokens.to(self.decoder_device, dtype=torch.long)
        if len(yseq) < 2:
            return tuple()
        decoder_input = yseq[:-1].unsqueeze(0)
        targets = yseq[1:]
        length = decoder_input.size(1)
        target_mask = torch.tril(
            torch.ones((length, length), device=self.decoder_device, dtype=torch.bool)
        ).unsqueeze(0)
        logits, _ = self.model.decoder(
            decoder_input, target_mask, encoded.unsqueeze(0), None
        )
        log_probabilities = torch.log_softmax(logits.squeeze(0), dim=-1)
        selected = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        return group_word_certainties(
            [int(token) for token in targets.detach().cpu()],
            [float(value) for value in selected.detach().cpu()],
            self.token_list,
        )
