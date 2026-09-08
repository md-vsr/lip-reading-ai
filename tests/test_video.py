from __future__ import annotations

import numpy as np
import pytest
import torch
from fractions import Fraction
from types import SimpleNamespace as NS
from unittest.mock import Mock

from app.video import (
    FaceNotFoundError,
    MouthPreprocessor,
    read_video_frames,
    resample_frames,
)


def test_resample_30_to_25_fps() -> None:
    frames = np.zeros((30, 8, 8, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(30)
    output = resample_frames(frames, source_fps=30, target_fps=25)
    assert output.shape == (25, 8, 8, 3)
    assert output[0, 0, 0, 0] == 0
    assert output[-1, 0, 0, 0] <= 29


def test_resample_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="RGB video"):
        resample_frames(np.zeros((10, 8, 8), dtype=np.uint8), 25, 25)


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("which", ["source", "target"])
def test_resample_rejects_nonfinite_and_nonpositive_fps(fps, which):
    frames = np.zeros((10, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="positive and finite"):
        resample_frames(
            frames, fps if which == "source" else 25, fps if which == "target" else 25
        )


def test_resample_uses_nonuniform_pts_instead_of_average_frame_indices():
    frames = np.stack([np.full((2, 2, 3), value, np.uint8) for value in range(4)])
    result = resample_frames(
        frames, 25, 25, timestamps=[10, 10.04, 10.08, 10.4], duration_seconds=0.44
    )
    assert len(result) == 11
    assert result[3, 0, 0, 0] == 2  # .12 seconds is nearest the frame at .08
    assert result[8, 0, 0, 0] == 3  # .32 seconds is nearest the frame at .40


@pytest.mark.parametrize(
    "times", [[0, 0, 0.04], [0, 0.08, 0.04], [0, 0.04], [0, float("nan"), 0.08]]
)
def test_resample_rejects_invalid_timestamps(times):
    with pytest.raises(ValueError, match="timestamps"):
        resample_frames(np.zeros((3, 2, 2, 3), np.uint8), 25, timestamps=times)


def fake_preprocessor(detection=None):
    processor = MouthPreprocessor.__new__(MouthPreprocessor)
    processor.minimum_detection_rate = 0.5
    processor.maximum_missing_frames = 25
    processor._landmarks_detector = NS(
        full_range_detector="full",
        short_range_detector="short",
        detect=Mock(
            side_effect=detection or (lambda frames, _: [np.ones((4, 2))] * len(frames))
        ),
    )
    processor._video_process = lambda frames, _: np.zeros(
        (len(frames), 96, 96, 3), np.uint8
    )
    processor._video_transform = lambda tensor: torch.zeros(len(tensor), 1, 88, 88)
    return processor


def test_process_frames_actually_resamples_and_preserves_source_count():
    result = fake_preprocessor().process_frames(
        np.zeros((30, 8, 8, 3), np.uint8), source_fps=30
    )
    assert result.source_frame_count == 30
    assert result.processed_frame_count == 25
    assert result.source_fps == 30
    assert result.duration_seconds == 1


def test_sparse_detection_is_rejected_before_crop():
    processor = fake_preprocessor(
        lambda frames, _: [np.ones((4, 2))] + [None] * (len(frames) - 1)
    )
    processor._video_process = Mock(
        side_effect=AssertionError("must not interpolate a bad track")
    )
    with pytest.raises(FaceNotFoundError, match="stable face"):
        processor.process_frames(np.zeros((100, 8, 8, 3), np.uint8))


def test_long_gap_is_rejected_even_if_overall_detection_rate_is_high():
    good = np.ones((4, 2))
    processor = fake_preprocessor(
        lambda frames, _: [good] * 100 + [None] * 30 + [good] * 100
    )
    with pytest.raises(FaceNotFoundError):
        processor.process_frames(np.zeros((230, 8, 8, 3), np.uint8))


def test_short_range_detector_is_retried_when_full_range_track_is_poor():
    processor = fake_preprocessor(
        lambda frames, backend: (
            [None if backend == "full" else np.ones((4, 2))] * len(frames)
        )
    )
    result = processor.process_frames(np.zeros((25, 8, 8, 3), np.uint8))
    assert result.face_detection_rate == 1
    assert processor._landmarks_detector.detect.call_count == 2


def test_read_budget_is_checked_before_rgb_allocation(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    path.touch()
    frame = NS(
        time=0.0,
        duration=1,
        time_base=Fraction(1, 25),
        width=1920,
        height=1080,
        to_ndarray=Mock(side_effect=AssertionError("should not allocate RGB")),
    )
    container = Mock()
    container.__enter__ = Mock(return_value=container)
    container.__exit__ = Mock(return_value=False)
    container.streams = NS(video=[NS(average_rate=25)])
    container.decode = Mock(return_value=iter([frame]))
    monkeypatch.setattr("app.video.av.open", Mock(return_value=container))
    with pytest.raises(ValueError, match="memory budget"):
        read_video_frames(path, maximum_bytes=1000)
    frame.to_ndarray.assert_not_called()


def test_read_real_sample_preserves_pts_and_source_count():
    from app.config import DEFAULT_SAMPLE

    decoded = read_video_frames(DEFAULT_SAMPLE)
    assert len(decoded.frames) == 178
    np.testing.assert_allclose(decoded.timestamps, np.arange(178) / 25)
    assert decoded.duration_seconds == pytest.approx(7.12)


def test_long_clips_are_rejected_explicitly():
    with pytest.raises(ValueError, match="split"):
        resample_frames(np.zeros((426, 2, 2, 3), np.uint8), 25)
