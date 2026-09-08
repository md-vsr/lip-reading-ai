from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av

from app.config import DEFAULT_CHECKPOINT, DEFAULT_SAMPLE, MODEL_DIR


MODEL_MANIFEST = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
CHECKPOINT_URL = MODEL_MANIFEST["direct_download_url"]
CHECKPOINT_BYTES = MODEL_MANIFEST["bytes"]
CHECKPOINT_SHA256 = MODEL_MANIFEST["sha256"]
SAMPLE_GIF_URL = "https://download.pytorch.org/torchaudio/doc-assets/avsr/original.gif"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    *,
    expected_sha256: str | None = None,
    timeout: float = 30,
    retries: int = 2,
) -> None:
    if timeout <= 0 or retries < 0:
        raise ValueError("Download timeout must be positive and retries non-negative.")
    if (
        destination.is_file()
        and (expected_size is None or destination.stat().st_size == expected_size)
        and (expected_sha256 is None or _sha256(destination) == expected_sha256)
    ):
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "lip-reading-ai/0.1"})
    for attempt in range(retries + 1):
        try:
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                partial.open("wb") as output,
            ):
                total = int(response.headers.get("Content-Length", "0"))
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if expected_size is not None and downloaded > expected_size:
                        raise RuntimeError(
                            f"Download exceeds expected size for {destination.name}."
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    if total:
                        print(
                            f"\rDownloading {destination.name}: {downloaded / total:6.1%}",
                            end="",
                            flush=True,
                        )
            print()
            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(
                    f"Unexpected size for {destination.name}: {downloaded}; expected {expected_size}."
                )
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"SHA-256 mismatch for {destination.name}.")
            partial.replace(destination)
            return
        except (OSError, RuntimeError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 4))


def _gif_to_mp4(gif_path: Path, mp4_path: Path) -> None:
    if mp4_path.is_file():
        print(f"Already present: {mp4_path}")
        return
    with av.open(str(gif_path)) as source:
        frames = list(source.decode(video=0))
        if not frames:
            raise RuntimeError(f"Sample GIF has no frames: {gif_path}")
        output = av.open(str(mp4_path), mode="w")
        stream = output.add_stream("libx264", rate=25)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        for index, frame in enumerate(frames):
            # GIF timestamps use a different time base; create clean CFR frames so
            # those timestamps cannot leak into the MP4 muxer.
            output_frame = av.VideoFrame.from_ndarray(
                frame.to_ndarray(format="rgb24"), format="rgb24"
            )
            output_frame.pts = index
            output_frame.time_base = Fraction(1, 25)
            for packet in stream.encode(output_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the official Auto-AVSR checkpoint and test clip"
    )
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_only and args.sample_only:
        parser.error("Choose at most one of --checkpoint-only and --sample-only.")

    if not args.sample_only:
        _download(
            CHECKPOINT_URL,
            DEFAULT_CHECKPOINT,
            CHECKPOINT_BYTES,
            expected_sha256=CHECKPOINT_SHA256,
        )
        print(f"Checkpoint SHA-256: {_sha256(DEFAULT_CHECKPOINT)}")
    if not args.checkpoint_only:
        gif_path = DEFAULT_SAMPLE.with_suffix(".gif")
        _download(SAMPLE_GIF_URL, gif_path)
        _gif_to_mp4(gif_path, DEFAULT_SAMPLE)
        gif_path.unlink(missing_ok=True)
        print(f"Sample SHA-256: {_sha256(DEFAULT_SAMPLE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
