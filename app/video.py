from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import av
import numpy as np
import torch

from app.config import (
    MAX_RECOMMENDED_SECONDS,
    TARGET_FPS,
    THIRD_PARTY_ROOT,
    validate_model_source,
)
from app.landmarks import FaceLandmarksDetector
from app.timing import validate_fps


MAX_RGB_BYTES = 512 * 1024 * 1024


class FaceNotFoundError(RuntimeError):
    """Raised when no usable face is visible in a segment."""


@dataclass(frozen=True)
class DecodedVideo:
    frames: np.ndarray
    source_fps: float
    duration_seconds: float
    timestamps: np.ndarray | None = None


@dataclass(frozen=True)
class PreprocessedVideo:
    tensor: torch.Tensor
    source_fps: float
    processed_fps: float
    source_frame_count: int
    processed_frame_count: int
    face_detection_rate: float

    @property
    def duration_seconds(self) -> float:
        return self.processed_frame_count / self.processed_fps


def _add_autoavsr_to_path() -> None:
    validate_model_source()
    source = str(THIRD_PARTY_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def read_video_frames(
    path: str | Path,
    *,
    maximum_seconds: float = MAX_RECOMMENDED_SECONDS,
    maximum_bytes: int = MAX_RGB_BYTES,
) -> DecodedVideo:
    """Read a bounded clip, retaining PTS rather than assuming constant fps.

    Limits are checked before RGB conversion/stacking. Long clips must be split
    explicitly, rather than silently truncated or allowed to exhaust memory.
    """
    validate_fps(maximum_seconds)
    if maximum_bytes <= 0:
        raise ValueError("RGB memory budget must be positive.")
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    total_bytes = 0
    origin = None
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {video_path}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else TARGET_FPS
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = TARGET_FPS
        for frame in container.decode(stream):
            raw_time = frame.time
            if origin is None:
                origin = float(raw_time) if raw_time is not None else 0.0
            timestamp = (
                float(raw_time) - origin
                if raw_time is not None
                else (timestamps[-1] + 1 / source_fps if timestamps else 0.0)
            )
            if not np.isfinite(timestamp) or (
                timestamps and timestamp <= timestamps[-1]
            ):
                raise ValueError(
                    "Video timestamps must be finite and strictly increasing."
                )
            frame_duration = (
                float(frame.duration * frame.time_base)
                if frame.duration and frame.time_base
                else 1 / source_fps
            )
            duration = timestamp + frame_duration
            if duration > maximum_seconds + 1e-6:
                raise ValueError(
                    f"Video exceeds {maximum_seconds:g} seconds; split it into shorter clips."
                )
            total_bytes += frame.width * frame.height * 3
            if total_bytes > maximum_bytes:
                raise ValueError(
                    "Video exceeds the RGB memory budget; shorten or downscale the clip."
                )
            if frames and (frame.height, frame.width) != frames[0].shape[:2]:
                raise ValueError("Video frame dimensions changed within the clip.")
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)

    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    array = np.stack(frames)
    return DecodedVideo(
        frames=array,
        source_fps=source_fps,
        duration_seconds=duration,
        timestamps=np.asarray(timestamps),
    )


def resample_frames(
    frames: np.ndarray,
    source_fps: float,
    target_fps: float = TARGET_FPS,
    *,
    timestamps: Sequence[float] | None = None,
    duration_seconds: float | None = None,
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB video shaped [frames, height, width, 3].")
    if len(frames) == 0 or min(frames.shape[1:3]) == 0:
        raise ValueError("Cannot resample an empty video or empty frame.")
    validate_fps(source_fps)
    validate_fps(target_fps)
    if timestamps is not None:
        times = np.asarray(timestamps, dtype=float)
        if (
            times.shape != (len(frames),)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "Frame timestamps must match frames and be finite and strictly increasing."
            )
        times = times - times[0]
        duration = (
            duration_seconds
            if duration_seconds is not None
            else times[-1] + 1 / source_fps
        )
    else:
        times = None
        duration = (
            duration_seconds
            if duration_seconds is not None
            else len(frames) / source_fps
        )
    validate_fps(duration)
    if times is not None and duration <= times[-1]:
        raise ValueError("Video duration must extend beyond the final frame timestamp.")
    if duration > MAX_RECOMMENDED_SECONDS + 1e-6:
        raise ValueError(
            "Video exceeds the maximum window length; split it into shorter clips."
        )
    if (
        times is None
        and duration_seconds is None
        and abs(source_fps - target_fps) < 1e-3
    ):
        return frames

    count = duration * target_fps
    if not np.isfinite(count) or count > MAX_RGB_BYTES // frames[0].nbytes:
        raise ValueError("Resampled video exceeds the RGB memory budget.")
    target_count = max(1, int(round(count)))
    if target_count * frames[0].nbytes > MAX_RGB_BYTES:
        raise ValueError("Resampled video exceeds the RGB memory budget.")
    targets = np.arange(target_count, dtype=np.float64) / target_fps
    if times is None:
        source_indices = np.rint(targets * source_fps).astype(np.int64)
    else:
        right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        source_indices = np.where(
            targets - times[left] < times[right] - targets, left, right
        )
    source_indices = np.clip(source_indices, 0, len(frames) - 1)
    return frames[source_indices]


class MouthPreprocessor:
    """Run Auto-AVSR's official MediaPipe alignment and test transform."""

    def __init__(
        self, minimum_detection_rate: float = 0.5, maximum_missing_seconds: float = 1.0
    ) -> None:
        if (
            not np.isfinite(minimum_detection_rate)
            or not 0 < minimum_detection_rate <= 1
        ):
            raise ValueError("Minimum detection rate must be in (0, 1].")
        validate_fps(maximum_missing_seconds)
        _add_autoavsr_to_path()
        from datamodule.transforms import VideoTransform
        from preparation.detectors.mediapipe.video_process import VideoProcess

        self._video_process = VideoProcess(convert_gray=False)
        self._video_transform = VideoTransform(subset="test")
        self.minimum_detection_rate = minimum_detection_rate
        self.maximum_missing_frames = round(maximum_missing_seconds * TARGET_FPS)
        self._landmarks_detector = FaceLandmarksDetector()

    def close(self) -> None:
        self._landmarks_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def process_file(self, path: str | Path) -> PreprocessedVideo:
        decoded = read_video_frames(path)
        return self.process_frames(
            decoded.frames,
            source_fps=decoded.source_fps,
            timestamps=decoded.timestamps,
            duration_seconds=decoded.duration_seconds,
        )

    def process_frames(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        source_fps: float = TARGET_FPS,
        *,
        timestamps: Sequence[float] | None = None,
        duration_seconds: float | None = None,
    ) -> PreprocessedVideo:
        # Check the budget before stacking a sequence of independent RGB arrays.
        if sum(np.asarray(frame).nbytes for frame in frames) > MAX_RGB_BYTES:
            raise ValueError("Video exceeds the RGB memory budget.")
        frame_array = np.asarray(frames, dtype=np.uint8)
        if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise ValueError("Expected RGB frames shaped [frames, height, width, 3].")
        if len(frame_array) < 3:
            raise ValueError(
                "At least three video frames are required for lip reading."
            )
        source_frame_count = len(frame_array)
        frame_array = resample_frames(
            frame_array,
            source_fps,
            TARGET_FPS,
            timestamps=timestamps,
            duration_seconds=duration_seconds,
        )
        if len(frame_array) < 3:
            raise ValueError(
                "At least three resampled video frames are required for lip reading."
            )

        # This follows the official detector's full-range-first behavior while allowing
        # us to report how many frames were detected before interpolation.
        detector = self._landmarks_detector
        landmarks = detector.detect(frame_array, detector.full_range_detector)
        if not self._adequate_landmarks(landmarks):
            alternative = detector.detect(frame_array, detector.short_range_detector)
            if self._adequate_landmarks(alternative):
                landmarks = alternative
        detected = sum(item is not None for item in landmarks)
        if not self._adequate_landmarks(landmarks):
            raise FaceNotFoundError(
                "No sufficiently stable face track was detected. Use a front-facing, "
                "well-lit video with the speaker's full mouth visible."
            )

        mouth = self._video_process(frame_array, landmarks)
        if mouth is None or len(mouth) == 0:
            raise FaceNotFoundError(
                "A face was found, but a stable mouth crop could not be produced."
            )
        if mouth.shape[1:3] != (96, 96):
            raise RuntimeError(
                f"Unexpected mouth crop shape: {mouth.shape}; expected [T, 96, 96, 3]."
            )

        tensor = torch.from_numpy(np.ascontiguousarray(mouth)).permute(0, 3, 1, 2)
        tensor = self._video_transform(tensor).float().contiguous()
        return PreprocessedVideo(
            tensor=tensor,
            source_fps=source_fps,
            processed_fps=TARGET_FPS,
            source_frame_count=source_frame_count,
            processed_frame_count=len(tensor),
            face_detection_rate=detected / len(frame_array),
        )

    def _adequate_landmarks(self, landmarks) -> bool:
        if not landmarks:
            return False
        missing_run = longest = detected = 0
        for item in landmarks:
            if item is None:
                missing_run += 1
                longest = max(longest, missing_run)
            else:
                detected += 1
                missing_run = 0
        return (
            detected / len(landmarks) >= self.minimum_detection_rate
            and longest <= self.maximum_missing_frames
        )
