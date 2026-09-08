from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.activity import LipMotionDetector, SpeechWindowCollector
from app.camera import discover_macos_cameras, open_camera, resolve_camera
from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
from app.model import WordCertainty
from app.native_logging import with_filtered_native_diagnostics
from app.pipeline import PipelineResult, VisualSpeechPipeline
from app.timing import FrameClock


WINDOW_TITLE = "Visual-Only Assistive Captions"


@dataclass(frozen=True)
class ReadySegment:
    entry_id: int
    frames: tuple[np.ndarray, ...]
    motion_fraction: float
    ready_at: float
    last_motion_at: float

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self.frames)


class SegmentQueue:
    """Bound pending RGB data independently of the caption display history."""

    def __init__(self, maximum_bytes: int = 512 * 1024 * 1024):
        if maximum_bytes <= 0:
            raise ValueError("Queue budget must be positive.")
        self.maximum_bytes = maximum_bytes
        self.nbytes = 0
        self.dropped = 0
        self.items: deque[ReadySegment] = deque()

    def __bool__(self):
        return bool(self.items)

    def popleft(self):
        item = self.items.popleft()
        self.nbytes -= item.nbytes
        return item

    def append(self, item: ReadySegment) -> list[int]:
        dropped = []
        if item.nbytes > self.maximum_bytes:
            self.dropped += 1
            return [item.entry_id]
        while self.items and self.nbytes + item.nbytes > self.maximum_bytes:
            dropped.append(self.popleft().entry_id)
        self.items.append(item)
        self.nbytes += item.nbytes
        self.dropped += len(dropped)
        return dropped

    def discard_expired(self, history):
        while self.items and not history.contains(self.items[0].entry_id):
            self.popleft()
            self.dropped += 1


def _capture_rgb(frame: np.ndarray, maximum_width: int = 640) -> np.ndarray:
    """Bound camera RGB storage before collecting a speech window."""
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) <= 0
        or maximum_width <= 0
    ):
        raise ValueError(
            "Expected a non-empty three-channel frame and positive size limit."
        )
    longest = max(frame.shape[:2])
    if longest > maximum_width:
        height = max(1, round(frame.shape[0] * maximum_width / longest))
        width = max(1, round(frame.shape[1] * maximum_width / longest))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@dataclass
class CaptionEntry:
    entry_id: int
    started_at: float
    text: str = ""
    word_certainties: tuple[WordCertainty, ...] = tuple()
    pending: bool = True


class CaptionHistory:
    """Keep the three most recent recognition windows in display order."""

    def __init__(self, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("Caption history must contain at least one row.")
        self.limit = limit
        self.entries: list[CaptionEntry] = []
        self._next_entry_id = 0

    def start(self, started_at: float | None = None) -> int:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        self.entries.append(
            CaptionEntry(
                entry_id=entry_id,
                started_at=time.monotonic() if started_at is None else started_at,
            )
        )
        if len(self.entries) > self.limit:
            self.entries.pop(0)
        return entry_id

    def complete(
        self,
        entry_id: int,
        text: str,
        word_certainties: tuple[WordCertainty, ...] = tuple(),
    ) -> bool:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.text = text
                entry.word_certainties = word_certainties
                entry.pending = False
                return True
        return False

    def contains(self, entry_id: int) -> bool:
        return any(entry.entry_id == entry_id for entry in self.entries)

    def discard(self, entry_id: int) -> bool:
        for index, entry in enumerate(self.entries):
            if entry.entry_id == entry_id:
                self.entries.pop(index)
                return True
        return False

    def display_rows(self) -> tuple[CaptionEntry | None, ...]:
        """Bottom-align entries so each new caption visibly shifts older rows up."""
        empty_rows = (None,) * (self.limit - len(self.entries))
        return empty_rows + tuple(self.entries)


def _processing_placeholder(started_at: float, now: float) -> str:
    phase = int(max(0.0, now - started_at) * 2) % 3
    return "." * (phase + 1)


def _caption_has_enough_support(
    motion_fraction: float,
    decoding_score_per_token: float | None,
    word_certainties: tuple[WordCertainty, ...],
) -> bool:
    """Reject language-prior text when visual and decoder evidence are both weak."""
    if (
        not math.isfinite(motion_fraction)
        or not 0 <= motion_fraction <= 1
        or decoding_score_per_token is None
        or not math.isfinite(decoding_score_per_token)
        or not word_certainties
        or any(
            not math.isfinite(item.certainty) or not 0 <= item.certainty <= 1
            for item in word_certainties
        )
    ):
        return False
    average_certainty = sum(item.certainty for item in word_certainties) / len(
        word_certainties
    )
    # These conservative rejection floors are heuristics, not calibrated
    # correctness probabilities. Strong motion must not bypass all evidence.
    if decoding_score_per_token < -3.0 or average_certainty < 0.15:
        return False
    if motion_fraction >= 0.3:
        return True
    if motion_fraction < 0.15 or decoding_score_per_token is None:
        return False
    if not word_certainties or decoding_score_per_token < -1.0:
        return False
    return average_certainty >= 0.55


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _certainty_color(certainty: float) -> tuple[int, int, int]:
    """Return an OpenCV BGR color from red (0%) through yellow to green (100%)."""
    certainty = min(1.0, max(0.0, certainty))
    red = round(255 * min(1.0, 2.0 * (1.0 - certainty)))
    green = round(255 * min(1.0, 2.0 * certainty))
    return (0, green, red)


def _draw_word_certainties(
    frame: np.ndarray,
    word_certainties: tuple[WordCertainty, ...],
    left: int,
    word_baseline: int,
    percentage_baseline: int,
    available_width: int,
    preferred_word_scale: float = 0.68,
    thickness: int = 2,
) -> None:
    """Draw large words with smaller, aligned certainty percentages underneath."""
    if not word_certainties:
        return

    def measure(
        word_scale: float,
    ) -> tuple[float, list[tuple[int, int]], int, int]:
        percentage_scale = word_scale * 0.52
        widths: list[tuple[int, int]] = []
        for item in word_certainties:
            word_width = cv2.getTextSize(
                item.word, cv2.FONT_HERSHEY_SIMPLEX, word_scale, thickness
            )[0][0]
            percentage_width = cv2.getTextSize(
                f"{item.certainty:.0%}",
                cv2.FONT_HERSHEY_SIMPLEX,
                percentage_scale,
                1,
            )[0][0]
            widths.append((word_width, percentage_width))
        gap = max(5, round(12 * word_scale / preferred_word_scale))
        total_width = sum(max(pair) for pair in widths) + gap * (len(widths) - 1)
        return percentage_scale, widths, gap, total_width

    word_scale = preferred_word_scale
    percentage_scale, widths, gap, total_width = measure(word_scale)
    if total_width > available_width:
        word_scale *= available_width / total_width
        percentage_scale, widths, gap, total_width = measure(word_scale)

    x = left + max(0, (available_width - total_width) // 2)
    for item, (word_width, percentage_width) in zip(word_certainties, widths):
        column_width = max(word_width, percentage_width)
        word_x = x + (column_width - word_width) // 2
        percentage_x = x + (column_width - percentage_width) // 2
        _draw_text(
            frame,
            item.word,
            (word_x, word_baseline),
            word_scale,
            (255, 255, 255),
            thickness,
        )
        _draw_text(
            frame,
            f"{item.certainty:.0%}",
            (percentage_x, percentage_baseline),
            percentage_scale,
            _certainty_color(item.certainty),
            1,
        )
        x += column_width + gap


def _fit_text_scale(
    text: str, available_width: int, preferred: float = 0.62, minimum: float = 0.3
) -> float:
    text_width = cv2.getTextSize(
        " ".join(text.split()), cv2.FONT_HERSHEY_SIMPLEX, preferred, 2
    )[0][0]
    if text_width <= available_width or text_width == 0:
        return preferred
    return max(minimum, preferred * available_width / text_width)


def _build_display_canvas(
    frame: np.ndarray,
    display_width: int,
    header_height: int = 72,
    caption_panel_height: int = 250,
) -> tuple[np.ndarray, int, int]:
    """Place the camera between separate header and caption panels."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR camera frame shaped [height, width, 3].")
    if display_width <= 0 or header_height < 0 or caption_panel_height <= 0:
        raise ValueError("Display dimensions must be positive.")
    source_height, source_width = frame.shape[:2]
    camera_height = max(1, round(source_height * display_width / source_width))
    camera_view = cv2.resize(
        frame, (display_width, camera_height), interpolation=cv2.INTER_AREA
    )
    camera_top = header_height
    caption_panel_top = camera_top + camera_height
    display = np.full(
        (caption_panel_top + caption_panel_height, display_width, 3),
        15,
        dtype=np.uint8,
    )
    display[camera_top:caption_panel_top] = camera_view
    return display, camera_top, caption_panel_top


def _create_resizable_window(initial_display: np.ndarray) -> None:
    """Create a user-resizable window while preserving the rendered layout ratio."""
    height, width = initial_display.shape[:2]
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_TITLE, width, height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Short-window visual-only webcam captions"
    )
    parser.add_argument(
        "--camera",
        default="phone",
        help="Camera selector: 'phone' (default), 'built-in', or a numeric index",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="List macOS cameras and exit"
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=12.0,
        help="Maximum seconds per detected visible-speech window",
    )
    parser.add_argument(
        "--mouth-motion-threshold",
        type=float,
        default=0.0055,
        help="Normalized lip-motion threshold used to start recognition",
    )
    parser.add_argument(
        "--minimum-speech-pause-seconds",
        type=float,
        default=0.5,
        help="Earliest endpoint when the lips have clearly settled",
    )
    parser.add_argument(
        "--speech-pause-seconds",
        type=float,
        default=1.0,
        help="Longest quiet interval before finalizing the current sentence",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Width of the camera and separate caption panel",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--ctc-weight", type=float, default=0.1)
    parser.add_argument("--no-decoder-cache", action="store_true")
    return parser


@with_filtered_native_diagnostics
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cameras:
        devices = discover_macos_cameras()
        if not devices:
            print("No cameras reported by macOS.")
            return 0
        for device in devices:
            if device.is_phone_camera:
                label = "iPhone/Continuity camera"
            elif device.is_builtin_mac_camera:
                label = "built-in Mac camera"
            else:
                label = "external camera"
            print(f"{device.index}: {device.name} ({label})")
        return 0
    if (
        not math.isfinite(args.window_seconds)
        or args.window_seconds < 1.0
        or args.window_seconds > 16.0
    ):
        raise ValueError("Window length must be between 1 and 16 seconds.")
    if (
        not math.isfinite(args.mouth_motion_threshold)
        or args.mouth_motion_threshold <= 0
    ):
        raise ValueError("Mouth-motion threshold must be positive.")
    if (
        not math.isfinite(args.speech_pause_seconds)
        or args.speech_pause_seconds < 0.2
        or args.speech_pause_seconds > 2.0
    ):
        raise ValueError("Speech pause must be between 0.2 and 2 seconds.")
    if (
        not math.isfinite(args.minimum_speech_pause_seconds)
        or args.minimum_speech_pause_seconds < 0.2
        or args.minimum_speech_pause_seconds > args.speech_pause_seconds
    ):
        raise ValueError(
            "Minimum speech pause must be at least 0.2 seconds and cannot exceed "
            "the maximum speech pause."
        )
    if args.display_width < 640 or args.display_width > 1920:
        raise ValueError("Display width must be between 640 and 1920 pixels.")

    device = resolve_camera(args.camera)
    print(f"Selected camera {device.index}: {device.name}")
    print("Loading visual speech model...")
    resources = ExitStack()
    try:
        pipeline = resources.enter_context(
            VisualSpeechPipeline(
                args.checkpoint,
                args.device,
                args.beam_size,
                ctc_weight=args.ctc_weight,
                decoder_cache=not args.no_decoder_cache,
            )
        )
        camera, first_frame = open_camera(device, TARGET_FPS)
        resources.callback(camera.release)
        motion_detector = LipMotionDetector(
            min_motion_score=args.mouth_motion_threshold
        )
        resources.callback(motion_detector.close)
        resources.callback(cv2.destroyAllWindows)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-speech")
        # Drain the worker before closing its MediaPipe detector.
        resources.callback(executor.shutdown, wait=True, cancel_futures=True)
        initial_display, _, _ = _build_display_canvas(first_frame, args.display_width)
        _create_resizable_window(initial_display)
    except BaseException:
        resources.close()
        raise

    def new_collector():
        return SpeechWindowCollector(
            fps=TARGET_FPS,
            maximum_seconds=args.window_seconds,
            ending_silence_seconds=args.speech_pause_seconds,
            minimum_ending_silence_seconds=args.minimum_speech_pause_seconds,
        )

    speech_collector = new_collector()
    frame_clock = FrameClock(TARGET_FPS)
    last_observed_frame = None
    last_observation = None
    last_motion_at = time.monotonic()
    future_last_motion_at = 0.0
    queue_delay = 0.0
    future: Future[PipelineResult] | None = None
    future_entry_id: int | None = None
    future_motion_fraction = 0.0
    active_entry_id: int | None = None
    ready_segments = SegmentQueue()
    caption_history = CaptionHistory(limit=3)
    status = "WAITING FOR LIP MOVEMENT"
    face_visible = False
    lips_moving = False
    latency: float | None = None

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
            else:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError(f"{device.name} stopped returning frames.")
            # OpenCV does not expose a portable device capture timestamp. Use
            # frame delivery time, not the requested/possibly ignored camera fps.
            delivered_at = time.monotonic()
            rgb = _capture_rgb(frame)
            try:
                samples = frame_clock.update(rgb, delivered_at)
            except ValueError:
                if active_entry_id is not None:
                    caption_history.discard(active_entry_id)
                    active_entry_id = None
                speech_collector = new_collector()
                frame_clock = FrameClock(TARGET_FPS)
                motion_detector.reset()
                last_observed_frame = last_observation = None
                samples = frame_clock.update(rgb, delivered_at)
            for sample in samples:
                if sample.frame is last_observed_frame:
                    # A duplicated image carries no new measured motion.
                    motion = replace(last_observation, moving=False)
                else:
                    motion = motion_detector.observe(sample.frame)
                    last_observed_frame, last_observation = sample.frame, motion
                face_visible = motion.face_visible
                lips_moving = motion.active
                if motion.moving:
                    last_motion_at = sample.timestamp
                mouth_settled = (
                    motion.face_visible
                    and motion.motion_score <= motion.threshold * 0.35
                )
                collector_active = motion.active or (
                    speech_collector.capturing and motion.moving
                )
                window_update = speech_collector.update(
                    sample.frame,
                    collector_active,
                    mouth_settled,
                    mouth_moving=motion.moving,
                )
                if window_update.started:
                    if active_entry_id is not None:
                        raise RuntimeError(
                            "A new speech window started before the last one ended."
                        )
                    active_entry_id = caption_history.start()
                if window_update.completed_frames is not None:
                    if active_entry_id is None:
                        raise RuntimeError(
                            "A speech window ended without a caption row."
                        )
                    dropped = ready_segments.append(
                        ReadySegment(
                            active_entry_id,
                            window_update.completed_frames,
                            window_update.motion_fraction,
                            time.monotonic(),
                            last_motion_at,
                        )
                    )
                    for entry_id in dropped:
                        caption_history.discard(entry_id)
                    active_entry_id = None
                elif window_update.discarded:
                    if active_entry_id is None:
                        raise RuntimeError(
                            "A rejected speech window had no caption row."
                        )
                    caption_history.discard(active_entry_id)
                    active_entry_id = None

            if future is not None and future.done():
                if future_entry_id is None:
                    raise RuntimeError(
                        "A recognition task finished without a caption row."
                    )
                try:
                    result = future.result()
                    if _caption_has_enough_support(
                        future_motion_fraction,
                        result.recognition.decoding_score_per_token,
                        result.recognition.word_certainties,
                    ):
                        caption_history.complete(
                            future_entry_id,
                            result.transcription or "[No words decoded — try again]",
                            result.recognition.word_certainties,
                        )
                    else:
                        caption_history.discard(future_entry_id)
                    latency = time.monotonic() - future_last_motion_at
                except Exception as exc:
                    caption_history.complete(
                        future_entry_id, f"Could not transcribe: {exc}"
                    )
                future = None
                future_entry_id = None
                future_motion_fraction = 0.0

            ready_segments.discard_expired(caption_history)
            if future is None and ready_segments:
                segment = ready_segments.popleft()
                future_entry_id = segment.entry_id
                future_motion_fraction = segment.motion_fraction
                future_last_motion_at = segment.last_motion_at
                queue_delay = time.monotonic() - segment.ready_at
                future = executor.submit(pipeline.transcribe_frames, segment.frames)
                del segment

            if speech_collector.capturing and future is not None:
                status = "CAPTURING / PROCESSING"
            elif speech_collector.capturing:
                status = "CAPTURING VISIBLE SPEECH"
            elif future is not None or ready_segments:
                status = "PROCESSING VISUAL SPEECH"
            else:
                status = "WAITING FOR LIP MOVEMENT"

            display, _, panel_top = _build_display_canvas(frame, args.display_width)
            height, width = display.shape[:2]
            face_color = (80, 220, 100) if face_visible else (80, 180, 255)
            _draw_text(
                display,
                "ASSISTIVE CAPTIONING PROTOTYPE",
                (18, 30),
                0.65,
                (255, 255, 255),
            )
            _draw_text(display, status, (18, 62), 0.72, (80, 220, 255))
            _draw_text(
                display,
                (
                    "LIPS MOVING"
                    if lips_moving
                    else "FACE READY"
                    if face_visible
                    else "POSITION FACE TOWARD CAMERA"
                ),
                (max(18, width - 360), 31),
                0.55,
                (80, 220, 100) if lips_moving else face_color,
            )
            if latency is not None:
                _draw_text(
                    display,
                    f"Last latency: {latency:.1f}s",
                    (max(18, width - 270), 62),
                    0.5,
                    (220, 220, 220),
                    1,
                )

            row_area_top = panel_top + 8
            row_area_bottom = height - 34
            row_height = max(38, (row_area_bottom - row_area_top) // 3)
            now = time.monotonic()
            for row_index, entry in enumerate(caption_history.display_rows()):
                row_top = row_area_top + row_index * row_height
                baseline = row_top + row_height // 2 + 8
                if entry is None:
                    _draw_text(display, "—", (22, baseline), 0.55, (105, 105, 105), 1)
                else:
                    if entry.pending:
                        placeholder = _processing_placeholder(entry.started_at, now)
                        _draw_text(
                            display,
                            placeholder,
                            (22, baseline),
                            0.78,
                            (80, 220, 255),
                        )
                    else:
                        if entry.word_certainties:
                            _draw_word_certainties(
                                display,
                                entry.word_certainties,
                                left=22,
                                word_baseline=row_top + row_height // 2,
                                percentage_baseline=row_top + row_height - 9,
                                available_width=width - 44,
                            )
                        else:
                            scale = _fit_text_scale(entry.text, width - 44)
                            _draw_text(
                                display,
                                entry.text,
                                (22, baseline),
                                scale,
                                (255, 255, 255),
                                2,
                            )
                if row_index < caption_history.limit - 1:
                    divider_y = row_area_top + (row_index + 1) * row_height
                    cv2.line(
                        display,
                        (18, divider_y),
                        (width - 18, divider_y),
                        (70, 70, 70),
                        1,
                        cv2.LINE_AA,
                    )
            _draw_text(
                display,
                f"Q: quit | queue {queue_delay:.1f}s | dropped {ready_segments.dropped}",
                (22, height - 12),
                0.42,
                (180, 180, 180),
                1,
            )
            _draw_text(
                display,
                "WORD CERTAINTY: decoder estimate, not calibrated",
                (max(18, width - 390), height - 12),
                0.42,
                (120, 190, 255),
                1,
            )

            cv2.imshow(WINDOW_TITLE, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        resources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
