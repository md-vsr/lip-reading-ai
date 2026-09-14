from __future__ import annotations

import os

import pytest

from app.config import DEFAULT_CHECKPOINT, DEFAULT_SAMPLE
from app.pipeline import VisualSpeechPipeline


@pytest.mark.integration
@pytest.mark.skipif(
    not (DEFAULT_CHECKPOINT.is_file() and DEFAULT_SAMPLE.is_file()),
    reason="Run scripts/download_assets.py first",
)
def test_visual_only_pipeline_produces_text() -> None:
    with VisualSpeechPipeline(
        checkpoint=DEFAULT_CHECKPOINT,
        device=os.environ.get("VSR_TEST_DEVICE", "auto"),
        beam_size=1,
    ) as pipeline:
        result = pipeline.transcribe_file(DEFAULT_SAMPLE)
    assert result.transcription.strip()
    assert result.frames >= 3
    assert 0 < result.face_detection_rate <= 1
    assert result.recognition.video_seconds > 0
    assert result.recognition.inference_seconds > 0
    assert len(result.recognition.word_certainties) == len(result.transcription.split())
    assert all(0 <= item.certainty <= 1 for item in result.recognition.word_certainties)


@pytest.mark.integration
@pytest.mark.skipif(
    not (DEFAULT_CHECKPOINT.is_file() and DEFAULT_SAMPLE.is_file()),
    reason="Run scripts/download_assets.py first",
)
def test_checkpoint_cached_decoder_matches_reference() -> None:
    with VisualSpeechPipeline(
        checkpoint=DEFAULT_CHECKPOINT,
        device=os.environ.get("VSR_TEST_DEVICE", "auto"),
        beam_size=3,
    ) as pipeline:
        video = pipeline.preprocessor.process_file(DEFAULT_SAMPLE)
        recognizer = pipeline.recognizer
        cached = recognizer.transcribe(video.tensor)
        recognizer.cached_decoder = None
        recognizer.beam_search.scorers["decoder"] = recognizer.model.decoder
        recognizer.beam_search.full_scorers["decoder"] = recognizer.model.decoder
        recognizer.beam_search.nn_dict["decoder"] = recognizer.model.decoder
        reference = recognizer.transcribe(video.tensor)
        assert cached.text == reference.text
        assert [word.word for word in cached.word_certainties] == [
            word.word for word in reference.word_certainties
        ]
        assert [word.certainty for word in cached.word_certainties] == pytest.approx(
            [word.certainty for word in reference.word_certainties],
            abs=1e-4,
        )
