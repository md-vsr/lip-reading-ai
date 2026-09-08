from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_CHECKPOINT
from app.native_logging import with_filtered_native_diagnostics
from app.pipeline import VisualSpeechPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe visible English speech from a video with Auto-AVSR (audio is ignored)."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Auto-AVSR visual checkpoint",
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Decoder beam width (larger is slower)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    parser.add_argument("--ctc-weight", type=float, default=0.1)
    parser.add_argument(
        "--no-decoder-cache",
        action="store_true",
        help="Use the unchanged upstream decoder for comparison",
    )
    parser.add_argument("--no-word-certainty", action="store_true")
    return parser


@with_filtered_native_diagnostics
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with VisualSpeechPipeline(
        args.checkpoint,
        args.device,
        args.beam_size,
        ctc_weight=args.ctc_weight,
        decoder_cache=not args.no_decoder_cache,
        word_certainty=not args.no_word_certainty,
    ) as pipeline:
        result = pipeline.transcribe_file(args.video)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    recognition = result.recognition
    print("\nVISUAL-ONLY TRANSCRIPTION")
    print(result.transcription or "[No words decoded]")
    print("\nDiagnostics")
    print(f"  Device: {recognition.device}")
    print(f"  Face detections: {result.face_detection_rate:.0%} of frames")
    print(
        f"  Video: {recognition.video_seconds:.2f}s ({result.frames} frames at 25 fps)"
    )
    print(f"  Model load: {recognition.model_load_seconds:.2f}s")
    print(f"  Mouth preprocessing: {result.preprocessing_seconds:.2f}s")
    print(f"  Inference: {recognition.inference_seconds:.2f}s")
    print(f"  Inference-only real-time factor: {recognition.real_time_factor:.2f}x")
    print(
        f"  Preprocessing + inference real-time factor: {result.compute_real_time_factor:.2f}x"
    )
    print(f"  Average process CPU: {recognition.average_process_cpu_percent:.0f}%")
    print(f"  Process memory: {recognition.memory_rss_mb:.0f} MB RSS")
    if recognition.word_certainties:
        print("\nWord certainty estimates (uncalibrated)")
        print(
            "  "
            + "  ".join(
                f"{item.word} [{item.certainty:.0%}]"
                for item in recognition.word_certainties
            )
        )
    if recognition.decoding_score_per_token is not None:
        print(
            "  Decoder score/token: "
            f"{recognition.decoding_score_per_token:.3f} (relative score, not calibrated confidence)"
        )
    print("  Audio use: disabled; no audio stream is decoded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
