from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from app.pipeline import PipelineResult, VisualSpeechPipeline


def test_preprocessor_is_closed_if_model_loading_fails(monkeypatch):
    processor = Mock()
    monkeypatch.setattr("app.pipeline.MouthPreprocessor", Mock(return_value=processor))
    monkeypatch.setattr(
        "app.pipeline.AutoAVSRRecognizer",
        Mock(side_effect=RuntimeError("checkpoint failure")),
    )
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        VisualSpeechPipeline()
    processor.close.assert_called_once()


def test_context_closes_preprocessor_on_processing_failure(monkeypatch):
    processor = Mock()
    processor.process_file.side_effect = ValueError("bad input")
    monkeypatch.setattr("app.pipeline.MouthPreprocessor", Mock(return_value=processor))
    monkeypatch.setattr("app.pipeline.AutoAVSRRecognizer", Mock())
    with pytest.raises(ValueError):
        with VisualSpeechPipeline() as pipeline:
            pipeline.transcribe_file("bad.mp4")
    processor.close.assert_called_once()


def test_compute_rtf_includes_preprocessing():
    result = PipelineResult(
        "test", NS(inference_seconds=2.0, video_seconds=5.0), 1.0, 1.0, 125
    )
    assert result.compute_seconds == 3.0
    assert result.compute_real_time_factor == 0.6
