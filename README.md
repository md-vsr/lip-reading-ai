# Lip-Reading AI

A working local visual speech recognition (VSR) prototype for assistive captioning. It
turns visible English speech into text and never uses the input audio stream. It supports:

- a normal video file to predicted text;
- a webcam UI that waits for sustained lip movement before capturing speech;
- Apple Metal acceleration with a CPU decoder compatibility path;
- per-word certainty estimates plus latency, real-time factor, CPU, and memory diagnostics.

This is an assistive captioning prototype, not a guarantee of correct transcription.

## Verified status on this machine

The end-to-end path was executed on an Apple M3 Mac with 24 GB RAM, macOS 26.3,
Python 3.11.15, PyTorch 2.7.1, and Metal available.

The included 7.12-second test MP4 has **no audio stream**. The program detected the
face in 90.4% of its frames, produced the expected 178×1×88×88 visual tensor, loaded
the real checkpoint, and decoded:

> COMPLETELY A CONSTRAINED ENVIRONMENT WHERE WE HAVE LARGE CHANGES IN NETBALLS AND

That is a model prediction, not reference ground truth. It proves that the real visual
pipeline executes; it does not establish accuracy on this particular clip.

| Measurement | Observed value |
| --- | ---: |
| Model load | 3.27 s |
| Face/mouth preprocessing | 1.66 s |
| Beam-10 inference | 3.73 s |
| Processed video | 7.12 s / 178 frames |
| Real-time factor | 0.52× |
| Average process CPU during inference | 93% |
| Process memory after inference | 988 MB RSS |

The values are one local run, not a benchmark guarantee. GPU utilization was not
recorded because macOS does not expose a suitable unprivileged per-process Metal metric.

## Model choice

The selected model is the maintained
[Auto-AVSR repository](https://github.com/mpc001/auto_avsr), pinned as a Git submodule
at commit `182b62837773ab01052d4ac21ef1d2203ea7d267`. The upstream repository reduced
dependencies in January 2025, supports an MPS preprocessing option, and publishes a
ready English visual-only model. Its code is Apache-2.0 licensed.

The checkpoint is `vsr_trlrs2lrs3vox2avsp_base.pth`:

- 250 million parameters and 1,001,892,616 bytes;
- SHA-256 `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`;
- trained on 3,291 hours associated with LRS2, LRS3, VoxCeleb2, and AVSpeech;
- upstream reports 20.3% word error rate on the LRS3 test set.

Those are upstream benchmark figures, not results reproduced by this project. The
upstream repository also warns that pretrained weights can carry terms derived from
their training datasets. Treat this prototype as research/evaluation software until the
relevant dataset and model-use rights have been confirmed.

[Auto-AVSR paper](https://arxiv.org/abs/2303.14307) ·
[model metadata](models/manifest.json) ·
[upstream license](third_party/auto_avsr/LICENSE)

### Why not AV-HuBERT?

The official AV-HuBERT repository was archived in September 2024 and its supported
setup uses Python 3.8 plus an older Fairseq stack. The PyTorch real-time AVSR example is
useful training code, but it does not publish a ready visual-only checkpoint. Auto-AVSR
was therefore the smallest compatibility risk for a working visual-only prototype on
this Apple-silicon machine.

## Architecture

```text
video stream only
  → decode RGB frames and resample to 25 fps
  → MediaPipe face detection (eyes, nose, mouth keypoints)
  → temporal landmark interpolation and affine face alignment
  → 96×96 mouth region
  → center crop to 88×88, grayscale, normalize (mean 0.421 / std 0.165)
  → 3D convolution + ResNet-18 visual frontend (512-D per frame)
  → linear projection to 768-D
  → 12-block Conformer encoder
  → 6-block Transformer decoder + joint CTC/attention beam search
  → 5,000-unit SentencePiece text
```

On MPS, the expensive visual frontend and Conformer run on Metal. The legacy ESPnet
CTC prefix scorer creates CPU index tensors, so decoding runs on CPU. Spatial-only
pooling uses an equivalent Metal-supported 2D operation. An inference adapter caches
decoder self-attention and encoder-memory K/V without modifying checkpoint weights
or third-party source; `--no-decoder-cache` selects the upstream reference decoder.

## Setup

The project uses `uv` and an isolated Python 3.11 environment. Nothing is installed
globally.

```bash
git submodule update --init --recursive
uv sync
.venv/bin/python scripts/fix_mediapipe_wheel.py
uv run --no-sync python scripts/system_check.py
uv run --no-sync python scripts/download_assets.py
```

The small metadata-fix step is needed because MediaPipe 0.10.21's official macOS
`universal2` wheel contains native arm64 binaries but incorrectly labels itself x86_64
internally. The script changes that installed metadata tag only.

## Video inference

```bash
uv run --no-sync python app/inference.py --video samples/test.mp4
uv run --no-sync python app/inference.py --video /absolute/path/to/video.mp4 --json
```

Useful options:

```text
--device auto|mps|cpu|cuda
--beam-size N                 # default 10; larger is slower
--ctc-weight FLOAT            # default 0.1; valid range 0..1
--no-decoder-cache            # reference upstream decoding path
--no-word-certainty           # omit word scoring (file CLI only)
--checkpoint /path/model.pth
--json
```

Input should show one mostly front-facing English speaker. The model was trained around
25 fps; other frame rates are resampled using PTS when available, including
variable-frame-rate inputs. File clips are limited to 16 seconds and a 512 MiB raw
RGB budget, checked during decoding before stacking. Longer/high-resolution clips
must be explicitly split or downscaled; clips are never silently truncated. The
application reports missing files, empty videos, unstable face tracks, invalid
devices, and unexpected crop shapes directly.

## Webcam captions

Camera delivery timestamps are resampled onto a 25 fps timeline rather than trusting
the requested camera fps. Gaps over 0.5 seconds discard the active window and reset
tracking. RGB capture buffers are downscaled to a maximum dimension of 640 pixels
(the displayed image is unchanged); pending work has a separate 512 MiB budget.
The UI reports dropped pending windows and queue delay. Caption latency is measured
from the last detected lip movement, including endpoint and queue waiting.

First grant camera access to the terminal or Codex app in **System Settings → Privacy &
Security → Camera**, then run:

```bash
uv run --no-sync python app/webcam.py
```

The default `--camera phone` selector chooses the iPhone/Continuity camera by name. To
inspect the mapping or switch back to the Mac's built-in camera:

```bash
uv run --no-sync python app/webcam.py --list-cameras
uv run --no-sync python app/webcam.py --camera phone
uv run --no-sync python app/webcam.py --camera built-in
uv run --no-sync python app/webcam.py --camera 1  # explicit index, if desired
```

Other options include `--window-seconds 12`, `--beam-size 1`, `--display-width 960`,
`--minimum-speech-pause-seconds 0.5`, `--speech-pause-seconds 1.0`, and the advanced
`--mouth-motion-threshold 0.0055` sensitivity control. Frames are kept in
memory only and are not written to disk. While the face is idle, the UI waits without
creating a row or asking the decoder for text. Sustained normalized lip movement starts
a speech window with `.`, `..`, and `...`. Endpointing combines elapsed quiet time with
the live lip-motion score: clearly settled lips close the window after 0.5 seconds,
while uncertain low movement gets up to 1 second. The 12-second maximum window remains
a final safety limit. Any renewed lip movement immediately cancels a pending endpoint.
The decoded text then replaces that placeholder. The newest placeholder is always on
the bottom row, and every older caption visibly shifts up one row when the next speech
window starts.
Before decoding, the app rejects windows without at least 0.2 seconds of measured lip
motion and a meaningful motion fraction. Marginal-motion windows are displayed only
when the decoder evidence is also strong. This suppresses generic language-model phrases
that can otherwise appear when no one is speaking without blacklisting specific text.
The UI keeps the latest three windows in a dedicated panel below the unobstructed camera
view. Per-word estimates remain colored from red (low), through yellow, to green (high).
The separate header shows face/lip activity, processing state, last latency, and a
warning that model uncertainty is not calibrated. The window can be resized by dragging
its edges; the full camera and caption layout keeps its aspect ratio. Press `Q` or Escape
to quit. Press `C` to switch to the next available Mac, iPhone/Continuity, or external
camera without restarting the app. The current camera is shown in the header; an
in-progress speech window is safely cleared when the video source changes.
Known MediaPipe graph startup diagnostics are filtered from the terminal, while
unexpected native errors remain visible. On Apple Silicon, the model's spatial-only
pooling uses an equivalent Metal-supported 2D operation instead of falling back to CPU.

Hardware verification found `FaceTime HD Camera` and `iPhone Camera`; the named
selectors avoid depending on their numeric indexes.

## Tests

See [reliability and decoder-cache changes](docs/RELIABILITY_FIXES.md) for the
implementation scope, new options, test coverage, and remaining validation work.

```bash
uv run --no-sync pytest
```

The suite includes camera selection, speech activity, caption history, word certainty,
frame processing, and a real integration smoke test. The integration test loads the
downloaded checkpoint, preprocesses the no-audio MP4, executes inference, and requires
a non-empty transcription. It skips only when the separately downloaded model or sample
is absent.

## Project layout

```text
app/                       application, preprocessing, model wrapper, CLI, webcam UI
models/                    ignored checkpoint plus tracked provenance/checksum
outputs/                   ignored generated output
samples/                   no-audio verification fixture
scripts/                   asset download, system check, Apple wheel metadata fix
tests/                     unit and real integration smoke tests
third_party/auto_avsr/     pinned official source as a Git submodule
pyproject.toml / uv.lock   exact isolated dependencies
```

## Limitations

Lip reading is inherently ambiguous: several speech sounds look identical on the lips,
and language context influences decoding. Results can change substantially with
lighting, camera angle, face or mouth resolution, occlusion, facial hair, speaking
style, language, distance, and differences from the training data. This checkpoint is
English-only and its relative beam score is **not** a calibrated probability. Do not
present a prediction as guaranteed truth.

The webcam mode uses mouth-motion endpointing but is not truly continuous streaming.
Large facial movements, poor landmark tracking, or an unusually subtle speaking style
can cause a false start or a missed utterance. Long utterances are still divided at the
configured maximum window length, and the 250M-parameter model has a several-second cold
start. A dedicated visual voice-activity model and a smaller streaming/Core ML decoder
would be the next reliability improvements.

### Per-word certainty

Console, JSON, and webcam results include an estimate for every decoded word. Each
value is the geometric mean of the Transformer decoder probabilities for the
SentencePiece tokens forming that word. The webcam displays the words in large white
text and aligns each smaller percentage directly underneath its word. It colors each
percentage from red (low), through yellow, to green (high).
These values are useful for comparing words within a prediction, but they are not
calibrated probabilities of correctness and can be overconfident because language
context contributes to the decoder score.
