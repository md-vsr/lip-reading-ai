"""Face selection fixes kept outside the pinned third-party submodule."""

from __future__ import annotations

import numpy as np


TRACK_REACQUIRE_AFTER_MISSES = 5


def _box(detection) -> np.ndarray:
    box = detection.location_data.relative_bounding_box
    return np.array([box.xmin, box.ymin, box.width, box.height], dtype=float)


def select_face(detections, previous_box: np.ndarray | None = None):
    """Start on the largest face, then require overlap with the tracked face.

    A disjoint bystander is treated as missing, not silently substituted for the
    speaker. Tracking state is local to one clip and does not leak across calls.
    """
    candidates = []
    for item in detections:
        box = _box(item)
        if np.isfinite(box).all() and np.all(box[2:] > 0):
            candidates.append((item, box))
    if not candidates:
        return None
    if previous_box is None:
        return max(candidates, key=lambda pair: pair[1][2] * pair[1][3])

    def overlap(pair):
        box = pair[1]
        size = np.maximum(
            0,
            np.minimum(box[:2] + box[2:], previous_box[:2] + previous_box[2:])
            - np.maximum(box[:2], previous_box[:2]),
        )
        intersection = float(np.prod(size))
        union = float(np.prod(box[2:]) + np.prod(previous_box[2:]) - intersection)
        return intersection / union

    best = max(candidates, key=overlap)
    return best if overlap(best) > 0 else None


class FaceLandmarksDetector:
    """Official four-keypoint geometry with corrected, consistent face selection."""

    def __init__(self):
        import mediapipe as mp

        self.short_range_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        try:
            self.full_range_detector = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5, model_selection=1
            )
        except BaseException:
            self.short_range_detector.close()
            raise
        self._closed = False

    def detect(self, frames, detector):
        landmarks = []
        previous_box = None
        tracking_misses = 0
        for frame in frames:
            result = detector.process(frame)
            detections = result.detections or []
            selected = select_face(detections, previous_box)
            if selected is None:
                tracking_misses += 1
                if tracking_misses >= TRACK_REACQUIRE_AFTER_MISSES:
                    previous_box = None
                    selected = select_face(detections)
            if selected is None:
                landmarks.append(None)
                continue
            tracking_misses = 0
            detection, previous_box = selected
            height, width = frame.shape[:2]
            keypoints = detection.location_data.relative_keypoints
            points = np.array(
                [
                    [int(point.x * width), int(point.y * height)]
                    for point in keypoints[:4]
                ]
            )
            landmarks.append(points if points.shape == (4, 2) else None)
        return landmarks

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.full_range_detector.close()
            finally:
                self.short_range_detector.close()
