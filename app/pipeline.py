from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
from app.model import AutoAVSRRecognizer, RecognitionResult
from app.video import MouthPreprocessor, PreprocessedVideo


@dataclass(frozen=True)
class PipelineResult:
    transcription: str
    recognition: RecognitionResult
    preprocessing_seconds: float
    face_detection_rate: float
    frames: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["recognition"] = self.recognition.to_dict()
        result["compute_seconds"] = self.compute_seconds
        result["compute_real_time_factor"] = self.compute_real_time_factor
        return result

    @property
    def compute_seconds(self) -> float:
        return self.preprocessing_seconds + self.recognition.inference_seconds

    @property
    def compute_real_time_factor(self) -> float:
        return self.compute_seconds / self.recognition.video_seconds


class VisualSpeechPipeline:
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
        self.preprocessor = MouthPreprocessor()
        try:
            self.recognizer = AutoAVSRRecognizer(
                checkpoint,
                device,
                beam_size,
                ctc_weight=ctc_weight,
                decoder_cache=decoder_cache,
                word_certainty=word_certainty,
            )
        except BaseException:
            self.preprocessor.close()
            raise

    def close(self) -> None:
        self.preprocessor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def transcribe_file(self, path: str | Path) -> PipelineResult:
        started = time.perf_counter()
        video = self.preprocessor.process_file(path)
        preprocessing_seconds = time.perf_counter() - started
        return self._recognize(video, preprocessing_seconds)

    def transcribe_frames(
        self,
        frames: Sequence[np.ndarray],
        source_fps: float = TARGET_FPS,
        *,
        timestamps: Sequence[float] | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        video = self.preprocessor.process_frames(
            frames, source_fps=source_fps, timestamps=timestamps
        )
        preprocessing_seconds = time.perf_counter() - started
        return self._recognize(video, preprocessing_seconds)

    def _recognize(
        self, video: PreprocessedVideo, preprocessing_seconds: float
    ) -> PipelineResult:
        recognition = self.recognizer.transcribe(video.tensor, video.processed_fps)
        return PipelineResult(
            transcription=recognition.text,
            recognition=recognition,
            preprocessing_seconds=preprocessing_seconds,
            face_detection_rate=video.face_detection_rate,
            frames=video.processed_frame_count,
        )
