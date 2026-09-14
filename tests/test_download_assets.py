import hashlib
import io
from unittest.mock import Mock

import pytest

from app.config import DEFAULT_SAMPLE
from scripts.download_assets import _download, _gif_to_mp4, _video_has_frames


def response(payload):
    handle = io.BytesIO(payload)
    handle.headers = {"Content-Length": str(len(payload))}
    return handle


def test_same_size_bad_checkpoint_is_replaced_only_after_hash_verification(
    tmp_path, monkeypatch
):
    file = tmp_path / "checkpoint.pth"
    file.write_bytes(b"BAD!")
    request = Mock(return_value=response(b"GOOD"))
    monkeypatch.setattr("urllib.request.urlopen", request)
    _download(
        "https://example.invalid/model",
        file,
        4,
        expected_sha256=hashlib.sha256(b"GOOD").hexdigest(),
    )
    assert file.read_bytes() == b"GOOD"
    assert request.call_args.kwargs["timeout"] == 30


def test_good_checkpoint_is_not_redownloaded(tmp_path, monkeypatch):
    file = tmp_path / "checkpoint.pth"
    file.write_bytes(b"GOOD")
    request = Mock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr("urllib.request.urlopen", request)
    _download("unused", file, 4, expected_sha256=hashlib.sha256(b"GOOD").hexdigest())
    request.assert_not_called()


@pytest.mark.parametrize("payload", [b"BAD!", b"SHORT", b"X"])
def test_bad_download_never_replaces_existing_file(tmp_path, monkeypatch, payload):
    file = tmp_path / "checkpoint.pth"
    file.write_bytes(b"OLD")
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response(payload)))
    with pytest.raises(RuntimeError):
        _download(
            "https://example.invalid/model",
            file,
            4,
            expected_sha256=hashlib.sha256(b"GOOD").hexdigest(),
            retries=0,
        )
    assert file.read_bytes() == b"OLD"
    assert not file.with_suffix(".pth.part").exists()


def test_connection_errors_retry_with_finite_limit(tmp_path, monkeypatch):
    request = Mock(side_effect=[TimeoutError(), response(b"GOOD")])
    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("scripts.download_assets.time.sleep", lambda _: None)
    file = tmp_path / "checkpoint.pth"
    _download("https://example.invalid/model", file, 4, retries=1)
    assert request.call_count == 2
    assert file.read_bytes() == b"GOOD"


def test_gif_conversion_atomically_replaces_invalid_output(tmp_path):
    output = tmp_path / "sample.mp4"
    output.write_bytes(b"incomplete")

    _gif_to_mp4(DEFAULT_SAMPLE, output)

    assert _video_has_frames(output)
    assert not (tmp_path / "sample.part.mp4").exists()
