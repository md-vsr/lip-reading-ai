from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    model_id: str
    unique_id: str

    @property
    def is_builtin_mac_camera(self) -> bool:
        identity = f"{self.name} {self.model_id} {self.unique_id}".lower()
        excluded = ("iphone", "continuity", "desk view")
        preferred = ("facetime", "built-in", "macbook", "imac")
        return not any(token in identity for token in excluded) and any(
            token in identity for token in preferred
        )

    @property
    def is_phone_camera(self) -> bool:
        identity = f"{self.name} {self.model_id} {self.unique_id}".lower()
        return "iphone" in identity or "continuity" in identity


def discover_macos_cameras() -> list[CameraDevice]:
    if platform.system() != "Darwin":
        return []
    completed = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(completed.stdout)
    records = payload.get("SPCameraDataType", [])
    return [
        CameraDevice(
            index=index,
            name=record.get("_name", f"Camera {index}"),
            model_id=record.get("spcamera_model-id", ""),
            unique_id=record.get("spcamera_unique-id", ""),
        )
        for index, record in enumerate(records)
    ]


def resolve_camera(camera: str) -> CameraDevice:
    value = camera.strip().lower()
    builtin_aliases = {"built-in", "builtin", "mac"}
    phone_aliases = {"phone", "iphone", "continuity"}
    if value not in builtin_aliases | phone_aliases:
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError(
                "Camera must be 'phone', 'built-in', or a numeric index such as 0."
            ) from exc
        if index < 0:
            raise ValueError("Camera index cannot be negative.")
        known = {device.index: device for device in discover_macos_cameras()}
        return known.get(index, CameraDevice(index, f"Camera {index}", "", ""))

    if value in builtin_aliases and platform.system() != "Darwin":
        return CameraDevice(0, "Default camera", "", "")
    if value in phone_aliases and platform.system() != "Darwin":
        raise RuntimeError(
            "The phone camera selector requires macOS Continuity Camera."
        )

    devices = discover_macos_cameras()
    if value in phone_aliases:
        phone_cameras = [device for device in devices if device.is_phone_camera]
        if phone_cameras:
            return min(
                phone_cameras,
                key=lambda device: (
                    "desk view" in device.name.lower(),
                    "iphone" not in device.name.lower(),
                    device.index,
                ),
            )
        names = ", ".join(device.name for device in devices) or "none"
        raise RuntimeError(
            "No iPhone/Continuity camera was found. Available cameras: " + names
        )

    for device in devices:
        if device.is_builtin_mac_camera:
            return device
    names = ", ".join(device.name for device in devices) or "none"
    raise RuntimeError("No built-in Mac camera was found. Available cameras: " + names)


def open_camera(
    device: CameraDevice,
    fps: float,
    warmup_seconds: float = 5.0,
) -> tuple[cv2.VideoCapture, np.ndarray]:
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(warmup_seconds)
        or warmup_seconds <= 0
    ):
        raise ValueError("Camera fps and warmup duration must be positive and finite.")
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(device.index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open {device.name} (camera {device.index}). Check Camera privacy permission."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, fps)

    deadline = time.monotonic() + warmup_seconds
    while time.monotonic() < deadline:
        ok, frame = camera.read()
        if ok and frame is not None:
            return camera, frame
        time.sleep(0.1)

    camera.release()
    raise RuntimeError(
        f"{device.name} opened but returned no frames after {warmup_seconds:.0f} seconds. "
        "Close FaceTime/Zoom and check macOS Camera privacy permission."
    )
