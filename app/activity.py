from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


LIP_LANDMARK_INDICES = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
)


@dataclass(frozen=True)
class LipMotionObservation:
    face_visible: bool
    motion_score: float
    threshold: float
    moving: bool
    active: bool


def normalized_lip_shape(landmarks: np.ndarray) -> np.ndarray:
    """Normalize lip points for face translation, scale, and in-plane rotation."""
    points = np.asarray(landmarks, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) <= max(LIP_LANDMARK_INDICES)
    ):
        raise ValueError("Expected at least 416 two-dimensional face landmarks.")

    left_corner = points[61]
    right_corner = points[291]
    mouth_axis = right_corner - left_corner
    mouth_width = float(np.linalg.norm(mouth_axis))
    if mouth_width <= 1e-6:
        raise ValueError("Mouth corners are too close to measure lip motion.")

    center = (left_corner + right_corner) / 2
    angle = math.atan2(float(mouth_axis[1]), float(mouth_axis[0]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    lip_points = points[list(LIP_LANDMARK_INDICES)]
    return ((lip_points - center) @ rotation) / mouth_width


def accumulated_lip_motion_score(
    current_shape: np.ndarray, previous_shapes: Sequence[np.ndarray]
) -> float:
    """Measure both immediate and slower lip movement across recent frames."""
    if not previous_shapes:
        return 0.0
    distances = [
        float(np.mean(np.linalg.norm(current_shape - previous, axis=1)))
        for previous in previous_shapes
    ]
    return max(distances[-1], float(np.median(distances)))


def adaptive_motion_threshold(minimum: float, noise_floor: float) -> float:
    """Adapt to camera jitter without allowing the speech threshold to run away."""
    adaptive = noise_floor * 2.5 + 0.0005
    return min(minimum * 1.35, max(minimum, adaptive))


@dataclass(frozen=True)
class MotionGateState:
    threshold: float
    moving: bool
    active: bool


class MotionActivationGate:
    """Require repeated motion while ignoring landmark-tracker warm-up jitter."""

    def __init__(
        self,
        minimum_score: float,
        vote_window: int = 6,
        required_motion_frames: int = 2,
        warmup_frames: int = 12,
    ) -> None:
        if not math.isfinite(minimum_score) or minimum_score <= 0:
            raise ValueError("The mouth-motion threshold must be positive.")
        if not 1 <= required_motion_frames <= vote_window:
            raise ValueError("Required motion frames must fit inside the vote window.")
        if warmup_frames < 0:
            raise ValueError("Warm-up frames cannot be negative.")
        self.minimum_score = minimum_score
        self.required_motion_frames = required_motion_frames
        self.warmup_frames = warmup_frames
        self._recent_motion: deque[bool] = deque(maxlen=vote_window)
        self._face_frames = 0
        self._noise_floor = minimum_score / 5

    def update(self, score: float, face_visible: bool) -> MotionGateState:
        if not math.isfinite(score) or score < 0:
            raise ValueError("Motion score must be finite and non-negative.")
        threshold = adaptive_motion_threshold(self.minimum_score, self._noise_floor)
        if not face_visible:
            self._recent_motion.clear()
            self._face_frames = 0
            return MotionGateState(threshold, False, False)

        self._face_frames += 1
        if 0 < score < self.minimum_score:
            self._noise_floor = 0.98 * self._noise_floor + 0.02 * score
            threshold = adaptive_motion_threshold(self.minimum_score, self._noise_floor)
        moving = self._face_frames > self.warmup_frames and score >= threshold
        self._recent_motion.append(moving)
        active = sum(self._recent_motion) >= self.required_motion_frames
        return MotionGateState(threshold, moving, active)


class LipMotionDetector:
    """Detect sustained lip-shape changes with MediaPipe Face Mesh."""

    def __init__(
        self,
        min_motion_score: float = 0.0055,
        vote_window: int = 6,
        required_motion_frames: int = 2,
        maximum_input_width: int = 640,
    ) -> None:
        self.min_motion_score = min_motion_score
        self.maximum_input_width = maximum_input_width
        self._shape_history: deque[np.ndarray] = deque(maxlen=6)
        self._activation_gate = MotionActivationGate(
            minimum_score=min_motion_score,
            vote_window=vote_window,
            required_motion_frames=required_motion_frames,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def observe(self, rgb_frame: np.ndarray) -> LipMotionObservation:
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an RGB camera frame shaped [height, width, 3].")
        height, width = frame.shape[:2]
        if width > self.maximum_input_width:
            resized_height = max(1, round(height * self.maximum_input_width / width))
            frame = cv2.resize(
                frame,
                (self.maximum_input_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        result = self._face_mesh.process(frame)
        if not result.multi_face_landmarks:
            self._shape_history.clear()
            gate = self._activation_gate.update(0.0, False)
            return LipMotionObservation(
                False, 0.0, gate.threshold, gate.moving, gate.active
            )

        frame_height, frame_width = frame.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array(
            [
                (landmark.x * frame_width, landmark.y * frame_height)
                for landmark in landmarks
            ],
            dtype=np.float32,
        )
        shape = normalized_lip_shape(points)
        score = accumulated_lip_motion_score(shape, self._shape_history)
        self._shape_history.append(shape)
        gate = self._activation_gate.update(score, True)
        return LipMotionObservation(
            True, score, gate.threshold, gate.moving, gate.active
        )

    def close(self) -> None:
        self._face_mesh.close()

    def reset(self) -> None:
        """Discard stale temporal evidence after a capture interruption."""
        self._shape_history.clear()
        self._activation_gate.update(0.0, False)
        self._face_mesh.reset()


@dataclass(frozen=True)
class SpeechWindowUpdate:
    started: bool = False
    completed_frames: tuple[np.ndarray, ...] | None = None
    discarded: bool = False
    motion_fraction: float = 0.0


class SpeechWindowCollector:
    """Collect a bounded speech window with preroll and adaptive endpointing."""

    def __init__(
        self,
        fps: float,
        maximum_seconds: float,
        preroll_seconds: float = 0.35,
        ending_silence_seconds: float = 1.0,
        minimum_seconds: float = 0.8,
        minimum_ending_silence_seconds: float | None = None,
        settled_silence_seconds: float = 0.2,
        minimum_motion_seconds: float = 0.2,
        minimum_motion_fraction: float = 0.15,
    ) -> None:
        values = (
            fps,
            maximum_seconds,
            preroll_seconds,
            ending_silence_seconds,
            minimum_seconds,
            settled_silence_seconds,
            minimum_motion_seconds,
            minimum_motion_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Speech window settings must be finite.")
        if preroll_seconds < 0 or minimum_seconds <= 0:
            raise ValueError(
                "Preroll must be non-negative and minimum duration positive."
            )
        if fps <= 0 or maximum_seconds <= 0:
            raise ValueError("Frame rate and maximum window length must be positive.")
        if minimum_seconds > maximum_seconds:
            raise ValueError("Minimum window length cannot exceed the maximum.")
        if ending_silence_seconds <= 0 or settled_silence_seconds <= 0:
            raise ValueError("Silence durations must be positive.")
        if minimum_ending_silence_seconds is None:
            minimum_ending_silence_seconds = min(0.5, ending_silence_seconds)
        if (
            not math.isfinite(minimum_ending_silence_seconds)
            or minimum_ending_silence_seconds <= 0
        ):
            raise ValueError("Minimum ending silence must be positive.")
        if minimum_ending_silence_seconds > ending_silence_seconds:
            raise ValueError("Minimum ending silence cannot exceed the maximum.")
        if minimum_motion_seconds <= 0:
            raise ValueError("Minimum motion duration must be positive.")
        if not 0 < minimum_motion_fraction <= 1:
            raise ValueError("Minimum motion fraction must be between zero and one.")
        self.maximum_frames = max(3, round(fps * maximum_seconds))
        self.minimum_frames = max(3, round(fps * minimum_seconds))
        self.minimum_ending_silence_frames = max(
            1, round(fps * minimum_ending_silence_seconds)
        )
        self.ending_silence_frames = max(1, round(fps * ending_silence_seconds))
        self.settled_silence_frames = max(1, round(fps * settled_silence_seconds))
        self.minimum_motion_frames = max(1, round(fps * minimum_motion_seconds))
        self.minimum_motion_fraction = minimum_motion_fraction
        preroll_frames = max(1, round(fps * preroll_seconds))
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._motion_preroll: deque[bool] = deque(maxlen=preroll_frames)
        self._frames: list[np.ndarray] | None = None
        self._motion_flags: list[bool] | None = None
        self._silent_frames = 0
        self._settled_frames = 0

    @property
    def capturing(self) -> bool:
        return self._frames is not None

    def update(
        self,
        frame: np.ndarray,
        mouth_active: bool,
        mouth_settled: bool = False,
        mouth_moving: bool | None = None,
    ) -> SpeechWindowUpdate:
        moving = mouth_active if mouth_moving is None else mouth_moving
        if self._frames is None:
            self._preroll.append(frame)
            self._motion_preroll.append(moving)
            if not mouth_active:
                return SpeechWindowUpdate()
            self._frames = list(self._preroll)
            self._motion_flags = list(self._motion_preroll)
            self._preroll.clear()
            self._motion_preroll.clear()
            self._silent_frames = 0
            self._settled_frames = 0
            return SpeechWindowUpdate(started=True)

        self._frames.append(frame)
        if self._motion_flags is None:
            raise RuntimeError(
                "Motion evidence was not initialized for the speech window."
            )
        self._motion_flags.append(moving)
        if mouth_active:
            self._silent_frames = 0
            self._settled_frames = 0
        else:
            self._silent_frames += 1
            self._settled_frames = self._settled_frames + 1 if mouth_settled else 0
        reached_maximum = len(self._frames) >= self.maximum_frames
        reached_confident_silence = (
            self._silent_frames >= self.minimum_ending_silence_frames
            and self._settled_frames >= self.settled_silence_frames
        )
        reached_silence_limit = self._silent_frames >= self.ending_silence_frames
        reached_silence = len(self._frames) >= self.minimum_frames and (
            reached_confident_silence or reached_silence_limit
        )
        if not reached_maximum and not reached_silence:
            return SpeechWindowUpdate()

        completed = tuple(self._frames)
        motion_flags = tuple(self._motion_flags)
        motion_frames = sum(motion_flags)
        motion_fraction = motion_frames / len(motion_flags)
        self._frames = None
        self._motion_flags = None
        self._silent_frames = 0
        self._settled_frames = 0
        self._preroll.extend(completed[-self._preroll.maxlen :])
        self._motion_preroll.extend(motion_flags[-self._motion_preroll.maxlen :])
        enough_motion = (
            motion_frames >= self.minimum_motion_frames
            and motion_fraction >= self.minimum_motion_fraction
        )
        if not enough_motion:
            return SpeechWindowUpdate(
                discarded=True,
                motion_fraction=motion_fraction,
            )
        return SpeechWindowUpdate(
            completed_frames=completed,
            motion_fraction=motion_fraction,
        )
