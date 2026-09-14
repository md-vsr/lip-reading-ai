"""Run the recognizer itself with a small synthetic checkpoint and real scorers."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch

from app.decoding import IncrementalDecoderScorer  # initializes the pinned source path
from app.model import AutoAVSRRecognizer
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.scorers.ctc import CTCPrefixScorer


TOKENS = ["<blank>"] + [f"▁WORD{index}" for index in range(10)] + ["<eos>"]


class Frontend(torch.nn.Module):
    def forward(self, video):
        return video.mean(dim=(-2, -1)).expand(-1, -1, 8)


class Encoder(torch.nn.Module):
    def forward(self, features, mask):
        return features, mask


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = Frontend()
        self.proj_encoder = torch.nn.Identity()
        self.encoder = Encoder()
        self.decoder = TransformerDecoder(
            12, attention_dim=8, attention_heads=2, linear_units=16, num_blocks=2
        )
        self.ctc = CTC(12, 8, 0.1)
        self.odim = 12
        self.eos = 11


def make_beam(model, token_list, beam_size, ctc_weight):
    return BatchBeamSearch(
        scorers={"decoder": model.decoder, "ctc": CTCPrefixScorer(model.ctc, eos=11)},
        weights={"decoder": 1 - ctc_weight, "ctc": ctc_weight},
        beam_size=beam_size,
        vocab_size=12,
        sos=11,
        eos=11,
        pre_beam_score_key="decoder" if ctc_weight < 1 else None,
    )


@pytest.mark.parametrize("ctc_weight", [0.0, 0.1, 1.0])
def test_recognizer_cache_matches_reference_and_reports_consistent_timings(
    tmp_path, monkeypatch, ctc_weight
):
    torch.manual_seed(1)
    path = tmp_path / "small.pth"
    torch.save(SmallModel().state_dict(), path)
    transform = NS(
        post_process=lambda ids: (
            "".join(TOKENS[int(token)] for token in ids).replace("▁", " ").strip()
        )
    )
    fake_lightning = NS(
        ModelModule=lambda _: NS(
            model=SmallModel(), token_list=TOKENS, text_transform=transform
        ),
        get_beam_search_decoder=make_beam,
    )
    monkeypatch.setitem(sys.modules, "lightning", fake_lightning)
    video = torch.rand(8, 1, 88, 88)
    cached_model = AutoAVSRRecognizer(path, "cpu", 3, ctc_weight=ctc_weight)
    if ctc_weight < 1:
        assert isinstance(cached_model.cached_decoder, IncrementalDecoderScorer)
        monkeypatch.setattr(
            cached_model,
            "_estimate_word_certainties",
            lambda *_: pytest.fail("cache should avoid teacher forcing"),
        )
    cached = cached_model.transcribe(video)
    reference_model = AutoAVSRRecognizer(
        path, "cpu", 3, ctc_weight=ctc_weight, decoder_cache=False
    )
    reference = reference_model.transcribe(video)
    assert cached.text == reference.text
    assert [word.word for word in cached.word_certainties] == [
        word.word for word in reference.word_certainties
    ]
    assert [word.certainty for word in cached.word_certainties] == pytest.approx(
        [word.certainty for word in reference.word_certainties],
        abs=1e-5,
    )
    assert (
        cached.encoder_seconds + cached.decoder_seconds + cached.certainty_seconds
        == pytest.approx(cached.inference_seconds)
    )
    if cached_model.cached_decoder is not None:
        assert cached_model.cached_decoder._memory_kv is None
    cached_model.word_certainty = False
    assert cached_model.transcribe(video).word_certainties == ()


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_recognizer_validates_fps_before_model_execution(fps):
    model = AutoAVSRRecognizer.__new__(AutoAVSRRecognizer)
    with pytest.raises(ValueError, match="positive and finite"):
        model.transcribe(torch.zeros(3, 1, 88, 88), fps=fps)
