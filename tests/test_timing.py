import numpy as np
import pytest

from app.activity import SpeechWindowCollector
from app.timing import FrameClock


@pytest.mark.parametrize("fps", [15, 25, 30, 60])
def test_camera_delivery_rates_share_one_time_axis(fps):
    clock = FrameClock(25)
    collector = SpeechWindowCollector(fps=25, maximum_seconds=2)
    samples = []
    ended_at = None
    for index in range(2 * fps + 1):
        for sample in clock.update(
            np.full((2, 2, 3), index % 255, np.uint8), 1000 + index / fps
        ):
            samples.append(sample)
            result = collector.update(sample.frame, True)
            if result.completed_frames is not None and ended_at is None:
                ended_at = sample.timestamp
    assert len(samples) == 51
    np.testing.assert_allclose(
        [item.timestamp - 1000 for item in samples], np.arange(51) / 25
    )
    assert ended_at == pytest.approx(1000 + 49 / 25)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_clock_rejects_invalid_fps(bad):
    with pytest.raises(ValueError):
        FrameClock(bad)


@pytest.mark.parametrize("time", [1.0, 0.9, 1.6, float("nan"), float("inf")])
def test_clock_rejects_discontinuities(time):
    clock = FrameClock()
    frame = np.zeros((2, 2, 3), np.uint8)
    clock.update(frame, 1.0)
    with pytest.raises(ValueError):
        clock.update(frame, time)


def test_clock_never_allocates_unbounded_gap_filling():
    clock = FrameClock()
    frame = np.zeros((2, 2, 3), np.uint8)
    clock.update(frame, 0.0)
    with pytest.raises(ValueError, match="gap"):
        clock.update(frame, 100_000.0)


def test_clock_rejects_resolution_changes():
    clock = FrameClock()
    clock.update(np.zeros((2, 2, 3), np.uint8), 0.0)
    with pytest.raises(ValueError, match="dimensions"):
        clock.update(np.zeros((4, 4, 3), np.uint8), 0.04)
