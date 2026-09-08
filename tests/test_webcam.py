from __future__ import annotations

import numpy as np
import pytest

from app import webcam
from app.model import WordCertainty
from app.webcam import (
    CaptionHistory,
    _build_display_canvas,
    _caption_has_enough_support,
    _certainty_color,
    _create_resizable_window,
    _draw_word_certainties,
    _processing_placeholder,
    build_parser,
)


def test_phone_camera_is_the_webcam_default() -> None:
    assert build_parser().parse_args([]).camera == "phone"


def test_webcam_defaults_keep_complete_sentences_together() -> None:
    args = build_parser().parse_args([])

    assert args.window_seconds == 12.0
    assert args.minimum_speech_pause_seconds == 0.5
    assert args.speech_pause_seconds == 1.0


def test_camera_and_caption_panel_do_not_overlap() -> None:
    camera_frame = np.full((90, 160, 3), 220, dtype=np.uint8)

    display, camera_top, caption_panel_top = _build_display_canvas(
        camera_frame,
        display_width=320,
        header_height=20,
        caption_panel_height=60,
    )

    assert display.shape == (260, 320, 3)
    assert camera_top == 20
    assert caption_panel_top == 200
    assert np.all(display[:camera_top] == 15)
    assert np.all(display[camera_top:caption_panel_top] == 220)
    assert np.all(display[caption_panel_top:] == 15)


def test_caption_window_is_user_resizable(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        webcam.cv2,
        "namedWindow",
        lambda title, flags: calls.append(("named", title, flags)),
    )
    monkeypatch.setattr(
        webcam.cv2,
        "resizeWindow",
        lambda title, width, height: calls.append(("resize", title, width, height)),
    )
    display = np.zeros((700, 960, 3), dtype=np.uint8)

    _create_resizable_window(display)

    assert calls == [
        (
            "named",
            webcam.WINDOW_TITLE,
            webcam.cv2.WINDOW_NORMAL | webcam.cv2.WINDOW_KEEPRATIO,
        ),
        ("resize", webcam.WINDOW_TITLE, 960, 700),
    ]


def test_certainty_color_runs_from_red_through_yellow_to_green() -> None:
    assert _certainty_color(0.0) == (0, 0, 255)
    assert _certainty_color(0.5) == (0, 255, 255)
    assert _certainty_color(1.0) == (0, 255, 0)


def test_certainty_color_clamps_values_to_probability_range() -> None:
    assert _certainty_color(-1.0) == _certainty_color(0.0)
    assert _certainty_color(2.0) == _certainty_color(1.0)


def test_percentage_is_smaller_and_drawn_below_its_word(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int], float, tuple[int, int, int]]] = []

    def record_text(frame, text, origin, scale, color, thickness=2) -> None:
        calls.append((text, origin, scale, color))

    monkeypatch.setattr(webcam, "_draw_text", record_text)
    frame = np.zeros((80, 400, 3), dtype=np.uint8)
    certainty = WordCertainty("HELLO", 0.8, 1)

    _draw_word_certainties(
        frame,
        (certainty,),
        left=10,
        word_baseline=28,
        percentage_baseline=55,
        available_width=380,
    )

    word_call, percentage_call = calls
    assert word_call[0] == "HELLO"
    assert percentage_call[0] == "80%"
    assert percentage_call[1][1] > word_call[1][1]
    assert percentage_call[2] < word_call[2]
    assert word_call[3] == (255, 255, 255)
    assert percentage_call[3] == _certainty_color(0.8)


def test_processing_placeholder_cycles_between_one_and_three_dots() -> None:
    assert _processing_placeholder(10.0, 10.0) == "."
    assert _processing_placeholder(10.0, 10.5) == ".."
    assert _processing_placeholder(10.0, 11.0) == "..."
    assert _processing_placeholder(10.0, 11.5) == "."


def test_caption_history_scrolls_when_fourth_placeholder_starts() -> None:
    history = CaptionHistory(limit=3)
    first = history.start(started_at=1.0)
    history.complete(first, "FIRST")
    second = history.start(started_at=2.0)
    history.complete(second, "SECOND")
    third = history.start(started_at=3.0)

    fourth = history.start(started_at=4.0)

    assert [entry.entry_id for entry in history.entries] == [second, third, fourth]
    assert [entry.text for entry in history.entries] == ["SECOND", "", ""]
    assert [entry.pending for entry in history.entries] == [False, True, True]


def test_caption_rows_are_bottom_anchored_and_shift_up_each_time() -> None:
    history = CaptionHistory(limit=3)
    first = history.start(started_at=1.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        None,
        None,
        first,
    ]

    second = history.start(started_at=2.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        None,
        first,
        second,
    ]

    third = history.start(started_at=3.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        first,
        second,
        third,
    ]

    fourth = history.start(started_at=4.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        second,
        third,
        fourth,
    ]


def test_caption_result_replaces_its_placeholder() -> None:
    history = CaptionHistory(limit=3)
    entry_id = history.start(started_at=1.0)
    certainties = (WordCertainty("HELLO", 0.8, 1),)

    history.complete(entry_id, "HELLO", certainties)

    assert history.entries[0].text == "HELLO"
    assert history.entries[0].word_certainties == certainties
    assert history.entries[0].pending is False


def test_caption_history_can_remove_a_rejected_placeholder() -> None:
    history = CaptionHistory(limit=3)
    entry_id = history.start(started_at=1.0)

    assert history.discard(entry_id)
    assert history.entries == []


def test_weak_visual_and_decoder_evidence_rejects_default_phrases() -> None:
    certainties = (
        WordCertainty("AND", 0.52, 1),
        WordCertainty("VIDEO", 0.56, 1),
    )

    assert not _caption_has_enough_support(0.2, -1.42, certainties)


def test_strong_visual_evidence_keeps_uncertain_real_speech() -> None:
    certainties = (WordCertainty("WORD", 0.3, 1),)

    assert _caption_has_enough_support(0.8, -1.5, certainties)


@pytest.mark.parametrize(
    "score,certainty",
    [(-100.0, 0.9), (-0.1, 0.001), (float("nan"), 0.9), (-0.1, float("nan"))],
)
def test_strong_motion_does_not_bypass_invalid_or_extremely_weak_decoder_evidence(
    score, certainty
):
    assert not _caption_has_enough_support(
        0.8, score, (WordCertainty("WORD", certainty, 1),)
    )


def test_capture_downscales_landscape_and_portrait_and_converts_color():
    for shape, expected in [
        ((720, 1280, 3), (360, 640, 3)),
        ((1280, 720, 3), (640, 360, 3)),
    ]:
        frame = np.zeros(shape, np.uint8)
        frame[:, :, 0] = 255
        rgb = webcam._capture_rgb(frame)
        assert rgb.shape == expected
        assert rgb[0, 0].tolist() == [0, 0, 255]


def test_pending_queue_enforces_byte_budget_and_reports_drops():
    frame = np.zeros((4, 4, 3), np.uint8)
    queue = webcam.SegmentQueue(maximum_bytes=frame.nbytes * 2)
    assert not queue.append(webcam.ReadySegment(1, (frame,), 0.8, 1.0, 0.0))
    assert not queue.append(webcam.ReadySegment(2, (frame,), 0.8, 2.0, 1.0))
    assert queue.append(webcam.ReadySegment(3, (frame,), 0.8, 3.0, 2.0)) == [1]
    assert queue.dropped == 1
    assert queue.nbytes <= queue.maximum_bytes
    assert queue.popleft().entry_id == 2
    assert queue.popleft().entry_id == 3
    assert queue.nbytes == 0


def test_queue_discards_oversized_item_without_evicting_valid_queued_work():
    frame = np.zeros((4, 4, 3), np.uint8)
    queue = webcam.SegmentQueue(maximum_bytes=frame.nbytes)
    queue.append(webcam.ReadySegment(1, (frame,), 0.8, 1.0, 0.0))
    assert queue.append(webcam.ReadySegment(2, (frame, frame), 0.8, 1.0, 0.0)) == [2]
    assert queue.popleft().entry_id == 1


def test_queue_removes_expired_caption_work():
    frame = np.zeros((4, 4, 3), np.uint8)
    queue = webcam.SegmentQueue()
    history = CaptionHistory(limit=1)
    old = history.start()
    queue.append(webcam.ReadySegment(old, (frame,), 0.8, 1.0, 0.0))
    history.start()
    queue.discard_expired(history)
    assert not queue and queue.nbytes == 0 and queue.dropped == 1
