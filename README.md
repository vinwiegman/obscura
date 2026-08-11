# Obscura

Track-aware face anonymization for video datasets and CCTV footage. Runs
entirely on your machine — nothing is uploaded anywhere.

```bash
obscura run footage.mp4 -o footage.redacted.mp4
```

## Why not just blur every detection?

That is the obvious implementation, and it leaks.

Face detectors miss. They miss on motion blur, on profile angles, on partial
occlusion, on the two frames where someone walks behind a pillar. On stills a
95% recall detector is excellent. On a 30fps video it means roughly **one and a
half unblurred frames per second** — and a single clean frame is all it takes to
identify someone. Anonymization is not an average-case problem; it is a
worst-case one.

Obscura runs in two passes:

1. **Scan** — decode the video, detect and track faces, keep only box geometry.
2. **Heal** — repair the track timeline: interpolate across frames the detector
   dropped, extend each track backwards and forwards past its first and last
   sighting, and dilate every box.
3. **Render** — decode again and write the redacted output.

The reason this beats a streaming pipeline is *hindsight*. When the detector
loses a face at frame 400 and finds it again at frame 430, a single-pass tool has
no way to know those are the same face — at frame 400 the recovery hasn't
happened yet. Obscura has the whole timeline before it renders a single pixel,
so it interpolates the missing 29 boxes from both sides.

Only geometry is held between passes, so a 20-minute 1080p video costs a few MB
of RAM rather than 40 GB.

## Results

Measured on a synthetic clip with scripted detector dropouts (60 frames, one
face, seven frames deliberately missed) — the case is constructed, but the
ground truth is exact, so the leak count is a count and not an estimate. The
same numbers are asserted in `tests/test_pipeline.py`.

| mode | faces | leaked | leak rate | frames with a leak |
|---|---|---|---|---|
| per-frame (baseline) | 60 | 7 | 11.67% | 7 |
| tracked + healed | 60 | 0 | 0.00% | 0 |

Reproduce on your own footage with ground-truth boxes:

```bash
obscura bench clip.mp4 --annotations boxes.json
```

Annotations are JSON, frame index to `[x1, y1, x2, y2]` boxes:

```json
{"frames": {"0": [[120, 64, 180, 140]], "1": [[122, 65, 182, 141]]}}
```

`bench` scans once and scores both modes off the same detections, so the
difference is attributable to healing rather than to detector nondeterminism.

**On real footage these numbers will be worse than the table above, in both
columns.** The synthetic clip has one face on a clean background and scripted
misses. Crowds, small faces and heavy occlusion break tracking too. Publishing
a number from footage you annotated yourself is the honest version of this
table; treat the synthetic result as a demonstration that the mechanism works,
not as a performance claim.

## Install

```bash
pip install 'obscura[cpu]'    # CPU / Apple Silicon
pip install 'obscura[gpu]'    # NVIDIA CUDA
```

Detection and tracking come from [uniface](https://github.com/yakhyo/uniface),
which downloads ONNX weights on first run. Python 3.10+.

## Usage

Single file, defaults (RetinaFace, elliptical blur):

```bash
obscura run footage.mp4 -o clean.mp4
```

A whole directory, walked recursively:

```bash
obscura run ./raw-footage -o ./released
```

Heavier redaction for a public dataset release:

```bash
obscura run footage.mp4 -o clean.mp4 --method pixelate --margin 0.3 --conf 0.35
```

Reproduce the naive baseline, for comparison:

```bash
obscura run footage.mp4 -o leaky.mp4 --single-pass
```

### Options that matter

| Flag | Default | Notes |
|---|---|---|
| `--method` | `blur` | `blur`, `pixelate`, `fill`. Blur strength scales with box size — a fixed kernel that erases a distant face leaves a close-up one readable. |
| `--shape` | `ellipse` | Elliptical masks with a feathered edge look less like a redaction than hard rectangles. |
| `--margin` | `0.18` | Box dilation per side. Detectors box the face; hair and jaw are identifying too. |
| `--top-extra` | `0.25` | Extra headroom, since foreheads and hairlines sit above a tight bbox. |
| `--max-gap` | `45` | Longest detector gap to bridge. Beyond this the face has probably left frame and interpolating would smear a redaction across unrelated pixels. |
| `--lead` / `--trail` | `8` / `12` | Frames covered before the first and after the last detection. A face is detected once it turns far enough toward the camera — by then the earlier frames have already leaked. |
| `--conf` | `0.5` | Deliberately lower than you would pick for recognition. A false positive costs a blurred patch of wall; a false negative costs a face. |
| `--model` | `retinaface` | Also `scrfd`, `yolov5`, `yolov8`. |
| `--single-pass` | off | Baseline mode. Leaks. |

### Web UI

```bash
pip install 'obscura[web]'
obscura serve
```

Drag a video onto `http://127.0.0.1:8000`, watch progress, download the result.

> The UI has no authentication and runs jobs as the local user. It is built for
> one operator on `localhost`. Putting it on a network without an auth layer in
> front would let anyone upload arbitrary files and consume the machine.

### Docker

```bash
docker build -t obscura .
docker run --rm -v "$PWD:/data" obscura run /data/in.mp4 -o /data/out.mp4
```

Mount a volume at `/models` to keep downloaded weights between runs.

## Limitations

- **Video only, audio dropped by default.** OpenCV writes no audio track. Pass
  `--keep-audio` to remux the original with ffmpeg, if ffmpeg is installed.
- **Faces only.** Gait, tattoos, licence plates, badges and lanyards all survive
  this tool. Faces are the easy part of anonymization, not the whole of it.
- **Blur is not encryption.** Blur and pixelation are reversible in principle,
  and small pixelation blocks are recoverable in practice. For an adversarial
  threat model use `--method fill`.
- **Crowds degrade tracking.** Dense scenes produce identity switches; the
  redaction still lands, but healing across a switch can drag a box between two
  people.
- **Two decode passes** cost roughly 15% over a streaming design. That is the
  price of hindsight.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

The core — geometry, healing, redaction — has no uniface dependency and is
tested without models or network. `tests/test_pipeline.py` builds a synthetic
video and a detector that misses on cue, then asserts the leak count before and
after healing.

`tests/test_uniface_contract.py` is the exception: it runs against the real
uniface package and skips when it is absent. Offline tests against fakes cannot
notice uniface renaming a detector or moving the track id to a different column,
so those assumptions are pinned separately.

## Licence

MIT. Note that the model weights uniface downloads carry their own licences,
several of them non-commercial — check the specific detector before commercial
use.
