# Reliability and decoder-cache changes

## Implemented

- Keep video PTS, handle VFR nearest-neighbor resampling, and preserve original
  frame counts independently of the 25 fps output frame count. `process_frames`
  now actually resamples `source_fps`; it also accepts explicit timestamps.
- Reject non-finite/invalid frame rates and timestamps. Limit file clips to 16
  seconds and 512 MiB of decoded RGB before stacking. Reject oversized clips with
  an actionable error instead of silently truncating or exhausting memory.
- Sample webcam deliveries onto a fixed timeline. Reset the active window after a
  capture gap or resolution change. Duplicated images do not count as newly
  measured lip motion. OpenCV delivery time is a fallback, not a hardware timestamp.
- Bound the maximum camera buffer dimension to 640 pixels and pending queue data
  to 512 MiB. Evict oldest pending segments explicitly and count drops. The queue
  budget is not a bound on total process memory: active capture, in-flight work,
  stacking copies, model weights and activations also require memory.
- Correct largest-face area selection in a local adapter and prefer the overlapping
  face track thereafter. A disjoint bystander is not substituted for a lost speaker.
  The pinned Auto-AVSR submodule has not been edited.
- Reject face tracks below 50% detection or with more than 1 second of consecutive
  missing detections. Retry the short-range detector on a poor full-range track.
  Reject non-finite or extremely weak decoder evidence even with strong motion.
  These are conservative heuristics, not calibrated correctness probabilities;
  the detection limits are configurable in `MouthPreprocessor`.
- Add inference-only self-attention K/V caches per hypothesis and cross-attention
  K/V per utterance. Existing beam state selection handles branching/reordering.
  Recover decoder-only token probabilities along the chosen beam path instead of
  running an extra teacher-forced decoder pass. Handle upstream's forced EOS.
- Keep the original decoder with `--no-decoder-cache`; expose `--ctc-weight` and,
  for file inference, `--no-word-certainty`. A CTC weight of 1 uses the original
  teacher-forced word-scoring fallback because no attention scorer is in the beam.
- Avoid full activation permutations in MPS spatial pooling, and place model
  components directly on their final devices rather than moving the decoder to
  MPS and back to CPU during initialization.
- Check checkpoint size **and SHA-256** against `models/manifest.json`; reject
  same-size corruption, use bounded network timeouts/retries, and atomically
  publish only a verified download. Failed downloads preserve the old file.
- Close detector resources explicitly, including failed initialization and webcam
  teardown; drain the worker before closing the preprocessing detector.
- Add encoder/decoder/certainty timing fields, preprocessing-inclusive compute
  RTF, queue delay, and last-motion-to-result caption latency. Preserve the old
  inference-only RTF field for compatibility.

## Validation

```bash
git submodule update --init --recursive
uv sync --frozen
.venv/bin/python scripts/fix_mediapipe_wheel.py
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app scripts tests
```

Checkpoint-independent tests cover real upstream decoder/scorer modules, multiple
beam sizes and seeds, per-token scores, branching/reordering, forced EOS, resetting
between clips, and avoiding repeated K/V projections. A small synthetic checkpoint
also exercises `AutoAVSRRecognizer` itself with CTC weights 0, 0.1 and 1, including
word probabilities and timing consistency. These tests do not measure speech accuracy.

Other tests cover 15/25/30/60 fps deliveries, capture gaps, real sample PTS,
nonuniform timestamps, budgets checked before RGB conversion, quality gates,
download corruption/retries, resource cleanup, and the actual webcam control loop
with a fake camera/model. Spatial pooling is checked for odd dimensions,
non-contiguous input and MPS when available.

The supplied sample was preprocessed successfully after the changes, yielding
178 x 1 x 88 x 88 and a 90.45% detection rate. No physical webcam session was used
for this patch's validation.

The large checkpoint was unavailable during development because its download
connection timed out. Two integration tests are therefore conditional on the
checkpoint: non-empty real inference, and cached-vs-reference text/word-score
equivalence. Download the assets, then run:

```bash
.venv/bin/python scripts/download_assets.py --checkpoint-only
.venv/bin/python -m pytest -q -m integration
.venv/bin/python app/inference.py --video samples/test.mp4 --json
.venv/bin/python app/inference.py --video samples/test.mp4 --no-decoder-cache --json
```

No end-to-end speedup or WER improvement is claimed without this validation.

## Deliberately not included

- Truly streaming/long-video recognition, incremental ROI-only buffers, and a
  unified live Face Mesh/alignment detector. Buffer downscaling and hard bounds
  mitigate memory use but do not implement the full ROI-cache architecture.
- Retraining a visual speech-activity classifier, confidence calibration, or
  claiming that the new heuristic gates improve accuracy on every speaker.
- All-MPS CTC scoring, FP16/INT8 conversion, SDPA replacement of relative-position
  attention, Core ML export, or removal of the Lightning dependency.
- Replacing macOS camera enumeration with a hardware unique-ID capture backend.
- Changing checkpoint weights, the pinned submodule, or existing upstream branches.

These require additional platform testing or a representative labeled evaluation
set. The old README's M3 timing numbers are historical, not new patch benchmarks.
