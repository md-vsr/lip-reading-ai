"""Timestamp validation and bounded, online constant-frame-rate sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def validate_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rates must be positive and finite.")


@dataclass(frozen=True)
class TimedFrame:
    frame: np.ndarray
    timestamp: float


class FrameClock:
    """Online nearest-neighbor sampling at target fps, with bounded gap filling.

    A large capture gap is an error instead of fabricating seconds of motion.
    Do not feed synthetic duplicates to a motion detector as new observations.
    """

    def __init__(self, fps: float = 25.0, maximum_gap_seconds: float = 0.5):
        validate_fps(fps)
        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("Maximum capture gap must be positive and finite.")
        self.fps = fps
        self.maximum_gap_seconds = maximum_gap_seconds
        self._origin: float | None = None
        self._previous: TimedFrame | None = None
        self._index = 0

    def update(self, frame: np.ndarray, timestamp: float) -> tuple[TimedFrame, ...]:
        if not math.isfinite(timestamp):
            raise ValueError("Capture timestamp must be finite.")
        if self._previous is None:
            self._origin = timestamp
            self._previous = TimedFrame(frame, timestamp)
            self._index = 1
            return (self._previous,)
        elapsed = timestamp - self._previous.timestamp
        if elapsed <= 0:
            raise ValueError("Capture timestamps must be strictly increasing.")
        if elapsed > self.maximum_gap_seconds:
            raise ValueError(
                "Capture gap exceeded the limit; restart the speech window."
            )
        if frame.shape != self._previous.frame.shape:
            raise ValueError(
                "Camera frame dimensions changed; restart the speech window."
            )
        samples = []
        while self._origin + self._index / self.fps <= timestamp + 1e-9:
            target = self._origin + self._index / self.fps
            # Nearest observed frame on the two sides of a target timestamp.
            chosen = (
                frame
                if timestamp - target <= target - self._previous.timestamp
                else self._previous.frame
            )
            samples.append(TimedFrame(chosen, target))
            self._index += 1
        self._previous = TimedFrame(frame, timestamp)
        return tuple(samples)
