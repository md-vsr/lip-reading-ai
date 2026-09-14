from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.camera import CameraDevice, next_camera_device, resolve_camera


CAMERA_JSON = """{
  "SPCameraDataType": [
    {
      "_name": "FaceTime HD Camera",
      "spcamera_model-id": "FaceTime HD Camera",
      "spcamera_unique-id": "MAC-CAMERA"
    },
    {
      "_name": "iPhone Camera",
      "spcamera_model-id": "iPhone14,3",
      "spcamera_unique-id": "PHONE-CAMERA"
    }
  ]
}"""


@patch("app.camera.platform.system", return_value="Darwin")
@patch("app.camera.subprocess.run")
def test_builtin_selector_avoids_iphone(run: Mock, _system: Mock) -> None:
    run.return_value = Mock(stdout=CAMERA_JSON)
    selected = resolve_camera("built-in")
    assert selected.index == 0
    assert selected.name == "FaceTime HD Camera"
    assert selected.is_builtin_mac_camera


@patch("app.camera.platform.system", return_value="Darwin")
@patch("app.camera.subprocess.run")
def test_numeric_selector_can_still_choose_explicit_device(
    run: Mock, _system: Mock
) -> None:
    run.return_value = Mock(stdout=CAMERA_JSON)
    selected = resolve_camera("1")
    assert selected.index == 1
    assert selected.name == "iPhone Camera"
    assert not selected.is_builtin_mac_camera


@patch("app.camera.platform.system", return_value="Darwin")
@patch("app.camera.subprocess.run")
def test_phone_selector_chooses_iphone_by_name(run: Mock, _system: Mock) -> None:
    run.return_value = Mock(stdout=CAMERA_JSON)
    selected = resolve_camera("phone")
    assert selected.index == 1
    assert selected.name == "iPhone Camera"
    assert selected.is_phone_camera


def test_invalid_camera_selector() -> None:
    with pytest.raises(ValueError, match="phone"):
        resolve_camera("banana")


def test_builtin_camera_classification() -> None:
    assert CameraDevice(
        0, "FaceTime HD Camera", "FaceTime HD Camera", "x"
    ).is_builtin_mac_camera
    assert not CameraDevice(1, "iPhone Camera", "iPhone14,3", "y").is_builtin_mac_camera
    assert CameraDevice(1, "iPhone Camera", "iPhone14,3", "y").is_phone_camera


def test_next_camera_cycles_in_stable_index_order() -> None:
    built_in = CameraDevice(0, "FaceTime HD Camera", "Mac", "mac")
    phone = CameraDevice(1, "iPhone Camera", "iPhone", "phone")

    assert next_camera_device(built_in, [phone, built_in]) == phone
    assert next_camera_device(phone, [phone, built_in]) == built_in


def test_next_camera_requires_another_available_device() -> None:
    built_in = CameraDevice(0, "FaceTime HD Camera", "Mac", "mac")

    with pytest.raises(RuntimeError, match="No other camera"):
        next_camera_device(built_in, [built_in])
