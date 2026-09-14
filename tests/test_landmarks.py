from types import SimpleNamespace as NS

import numpy as np

from app.landmarks import FaceLandmarksDetector, select_face


def detection(x=0.1, y=0.1, width=0.6, height=0.6, marker=0.25):
    return NS(
        location_data=NS(
            relative_bounding_box=NS(xmin=x, ymin=y, width=width, height=height),
            relative_keypoints=[NS(x=marker, y=marker)] * 4,
        )
    )


def test_largest_face_is_independent_of_detection_order_and_position():
    large = detection(x=0.4, y=0.4, width=0.5, height=0.5)
    small = detection(x=0.0, y=0.0, width=0.2, height=0.2, marker=0.75)
    assert select_face([large, small])[0] is large
    assert select_face([small, large])[0] is large


def test_tracker_keeps_the_speaker_when_larger_bystander_appears():
    speaker = detection(x=0.1, y=0.1, width=0.2, height=0.2)
    stranger = detection(x=0.5, y=0.5, width=0.5, height=0.5)
    previous = select_face([speaker])[1]
    assert select_face([stranger, speaker], previous)[0] is speaker
    assert select_face([stranger], previous) is None


def test_invalid_boxes_are_ignored():
    assert select_face([detection(width=-1), detection(x=float("nan"))]) is None


def test_detector_uses_largest_face_four_keypoints():
    detector = FaceLandmarksDetector.__new__(FaceLandmarksDetector)
    backend = NS(
        process=lambda _: NS(
            detections=[detection(), detection(width=0.1, height=0.1, marker=0.75)]
        )
    )
    result = detector.detect([np.zeros((100, 100, 3), np.uint8)], backend)
    np.testing.assert_array_equal(result[0], np.full((4, 2), 25))
