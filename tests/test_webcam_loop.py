"""Exercise the real webcam control flow without a physical camera or model."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from app import webcam
from app.activity import LipMotionObservation
from app.camera import CameraDevice
from app.model import WordCertainty


@pytest.mark.parametrize("capture_gap", [False, True])
def test_webcam_samples_queues_and_cleans_up_resources(monkeypatch, capture_gap):
    now = [0.0]
    iterations = [0]
    camera_frame = np.zeros((24, 32, 3), np.uint8)
    camera = Mock()
    camera.read.return_value = (True, camera_frame)
    motion = Mock()
    motion.observe.return_value = LipMotionObservation(True, 0.1, 0.0055, True, True)
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    pipeline.transcribe_frames.return_value = NS(
        transcription="HELLO",
        preprocessing_seconds=0.01,
        recognition=NS(
            inference_seconds=0.01,
            decoding_score_per_token=-0.1,
            word_certainties=(WordCertainty("HELLO", 0.9, 1),),
        ),
    )
    monkeypatch.setattr(webcam, "VisualSpeechPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(webcam, "LipMotionDetector", Mock(return_value=motion))
    monkeypatch.setattr(
        webcam, "resolve_camera", lambda _: NS(index=0, name="Test camera")
    )
    monkeypatch.setattr(webcam, "open_camera", lambda *_: (camera, camera_frame))
    monkeypatch.setattr(webcam, "_create_resizable_window", lambda _: None)
    monkeypatch.setattr(webcam, "_draw_text", lambda *_, **__: None)
    monkeypatch.setattr(webcam.cv2, "imshow", lambda *_: None)
    close_ui = Mock()
    monkeypatch.setattr(webcam.cv2, "destroyAllWindows", close_ui)
    monkeypatch.setattr(webcam.time, "monotonic", lambda: now[0])

    def wait_key(_):
        iterations[0] += 1
        now[0] += 1.0 if capture_gap and iterations[0] == 8 else 0.04
        return ord("q") if iterations[0] == 40 else -1

    monkeypatch.setattr(webcam.cv2, "waitKey", wait_key)
    assert webcam.main.__wrapped__(["--camera", "0", "--window-seconds", "1"]) == 0
    assert pipeline.transcribe_frames.call_count >= 1
    assert len(pipeline.transcribe_frames.call_args.args[0]) == 25
    camera.release.assert_called_once()
    motion.close.assert_called_once()
    pipeline.__exit__.assert_called_once()
    close_ui.assert_called_once()
    assert motion.reset.call_count == int(capture_gap)


def test_model_resources_close_when_camera_initialization_fails(monkeypatch):
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    monkeypatch.setattr(webcam, "VisualSpeechPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(
        webcam, "resolve_camera", lambda _: NS(index=0, name="Test camera")
    )
    monkeypatch.setattr(
        webcam, "open_camera", Mock(side_effect=RuntimeError("no camera"))
    )
    with pytest.raises(RuntimeError, match="no camera"):
        webcam.main.__wrapped__(["--camera", "0"])
    pipeline.__exit__.assert_called_once()


def test_c_key_switches_camera_and_resets_capture_state(monkeypatch):
    now = [0.0]
    key_presses = iter((ord("c"), ord("q")))
    first_frame = np.zeros((24, 32, 3), np.uint8)
    second_frame = np.ones((24, 32, 3), np.uint8)
    first_camera = Mock()
    second_camera = Mock()
    motion = Mock()
    motion.observe.return_value = LipMotionObservation(
        True, 0.0, 0.0055, False, False
    )
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    built_in = CameraDevice(0, "FaceTime HD Camera", "Mac", "mac")
    phone = CameraDevice(1, "iPhone Camera", "iPhone", "phone")
    open_camera = Mock(
        side_effect=((first_camera, first_frame), (second_camera, second_frame))
    )

    monkeypatch.setattr(webcam, "VisualSpeechPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(webcam, "LipMotionDetector", Mock(return_value=motion))
    monkeypatch.setattr(webcam, "resolve_camera", lambda _: built_in)
    monkeypatch.setattr(webcam, "discover_macos_cameras", lambda: [built_in, phone])
    monkeypatch.setattr(webcam, "open_camera", open_camera)
    monkeypatch.setattr(webcam, "_create_resizable_window", lambda _: None)
    monkeypatch.setattr(webcam, "_draw_text", lambda *_, **__: None)
    monkeypatch.setattr(webcam.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(webcam.cv2, "destroyAllWindows", Mock())
    monkeypatch.setattr(webcam.cv2, "waitKey", lambda _: next(key_presses))
    monkeypatch.setattr(webcam.time, "monotonic", lambda: now[0])

    assert webcam.main.__wrapped__(["--camera", "0"]) == 0
    assert [call.args[0] for call in open_camera.call_args_list] == [built_in, phone]
    first_camera.release.assert_called_once()
    second_camera.release.assert_called_once()
    motion.reset.assert_called_once()
    pipeline.__exit__.assert_called_once()


def test_failed_camera_switch_restores_previous_camera(monkeypatch):
    key_presses = iter((ord("c"), ord("q")))
    frame = np.zeros((24, 32, 3), np.uint8)
    first_camera = Mock()
    restored_camera = Mock()
    motion = Mock()
    motion.observe.return_value = LipMotionObservation(
        True, 0.0, 0.0055, False, False
    )
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    built_in = CameraDevice(0, "FaceTime HD Camera", "Mac", "mac")
    phone = CameraDevice(1, "iPhone Camera", "iPhone", "phone")
    open_camera = Mock(
        side_effect=(
            (first_camera, frame),
            RuntimeError("phone unavailable"),
            (restored_camera, frame),
        )
    )

    monkeypatch.setattr(webcam, "VisualSpeechPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(webcam, "LipMotionDetector", Mock(return_value=motion))
    monkeypatch.setattr(webcam, "resolve_camera", lambda _: built_in)
    monkeypatch.setattr(webcam, "discover_macos_cameras", lambda: [built_in, phone])
    monkeypatch.setattr(webcam, "open_camera", open_camera)
    monkeypatch.setattr(webcam, "_create_resizable_window", lambda _: None)
    monkeypatch.setattr(webcam, "_draw_text", lambda *_, **__: None)
    monkeypatch.setattr(webcam.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(webcam.cv2, "destroyAllWindows", Mock())
    monkeypatch.setattr(webcam.cv2, "waitKey", lambda _: next(key_presses))

    assert webcam.main.__wrapped__(["--camera", "0"]) == 0
    assert [call.args[0] for call in open_camera.call_args_list] == [
        built_in,
        phone,
        built_in,
    ]
    first_camera.release.assert_called_once()
    restored_camera.release.assert_called_once()
    motion.reset.assert_called_once()
    pipeline.__exit__.assert_called_once()
