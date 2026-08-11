"""Leak-rate benchmark: healed coverage against the per-frame baseline.

A "leak" is a ground-truth face that the redaction failed to obscure. One leaked
frame in a 30fps video is enough to identify someone, so the metric that matters
is not mean IoU but the count of faces left readable.

Annotations are JSON, frame index to a list of ``[x1, y1, x2, y2]`` boxes::

    {"frames": {"0": [[120, 64, 180, 140]], "1": [[122, 65, 182, 141]]}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import video
from .config import DetectConfig, HealConfig, RunConfig, TrackConfig
from .detect import UnifaceDetector
from .geometry import Box
from .pipeline import scan
from .timeline import FrameIndex, index_from_detections
from .track import build as build_tracker


def load_annotations(path: Path) -> dict[int, list[Box]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    annotations: dict[int, list[Box]] = {}
    for key, boxes in frames.items():
        annotations[int(key)] = [Box(*(float(v) for v in box[:4])) for box in boxes]
    return annotations


def coverage(target: Box, boxes: list[Box]) -> float:
    """Fraction of ``target``'s area covered by the union of ``boxes``.

    Rasterized rather than computed analytically: redaction boxes overlap freely,
    and inclusion-exclusion over a dozen of them is easy to get subtly wrong.
    """
    x1, y1, x2, y2 = target.to_int()
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    mask = np.zeros((height, width), dtype=bool)

    for box in boxes:
        bx1, by1, bx2, by2 = box.to_int()
        ix1, iy1 = max(x1, bx1) - x1, max(y1, by1) - y1
        ix2, iy2 = min(x2, bx2) - x1, min(y2, by2) - y1
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True

    return float(mask.mean())


@dataclass(slots=True)
class ModeResult:
    name: str
    faces: int
    leaked_faces: int
    leaked_frames: int
    annotated_frames: int
    redacted_fraction: float
    """Mean fraction of each frame that gets obscured -- the cost side of the trade."""

    @property
    def leak_rate(self) -> float:
        return self.leaked_faces / self.faces if self.faces else 0.0

    @property
    def frame_leak_rate(self) -> float:
        return self.leaked_frames / self.annotated_frames if self.annotated_frames else 0.0


@dataclass(slots=True)
class BenchReport:
    source: Path
    n_frames: int
    n_tracks: int
    baseline: ModeResult
    healed: ModeResult

    def format(self) -> str:
        lines = [
            f"{self.source.name}: {self.n_frames} frames, {self.n_tracks} tracks",
            "",
            f"{'mode':<14}{'faces':>8}{'leaked':>9}{'leak rate':>12}"
            f"{'leaky frames':>15}{'frame area':>13}",
        ]
        for result in (self.baseline, self.healed):
            lines.append(
                f"{result.name:<14}{result.faces:>8}{result.leaked_faces:>9}"
                f"{result.leak_rate:>11.2%}{result.frame_leak_rate:>14.2%}"
                f"{result.redacted_fraction:>12.2%}"
            )

        before, after = self.baseline.leak_rate, self.healed.leak_rate
        if before > 0:
            lines += ["", f"healing removes {(before - after) / before:.1%} of leaked faces"]
        return "\n".join(lines)


def _evaluate(
    name: str,
    index: FrameIndex,
    annotations: dict[int, list[Box]],
    frame_area: float,
    threshold: float,
) -> ModeResult:
    faces = leaked_faces = leaked_frames = 0
    redacted = 0.0

    for boxes in index:
        redacted += sum(b.area for b in boxes) / frame_area if frame_area else 0.0

    for frame_index, truth in sorted(annotations.items()):
        boxes = index[frame_index] if 0 <= frame_index < len(index) else []
        leaks_here = 0
        for target in truth:
            faces += 1
            if coverage(target, boxes) < threshold:
                leaks_here += 1
        leaked_faces += leaks_here
        leaked_frames += 1 if leaks_here else 0

    return ModeResult(
        name=name,
        faces=faces,
        leaked_faces=leaked_faces,
        leaked_frames=leaked_frames,
        annotated_frames=len(annotations),
        # Boxes may overlap, so this slightly overstates area; it is a relative
        # comparison between two modes, not an absolute measurement.
        redacted_fraction=redacted / len(index) if index else 0.0,
    )


def run_benchmark(
    source: Path,
    annotations_path: Path,
    threshold: float = 0.9,
    model: str = "retinaface",
    conf: float = 0.5,
    gpu: bool = False,
) -> BenchReport:
    """Scan once, then score the healed and baseline indexes off the same detections.

    Sharing a single scan keeps the comparison honest: any difference comes from
    healing, not from detector nondeterminism between two runs.
    """
    annotations = load_annotations(annotations_path)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else None
    cfg = RunConfig(
        detect=DetectConfig(model=model, conf=conf, providers=providers),
        track=TrackConfig(),
        heal=HealConfig(),
    )

    meta = video.probe(source)
    result = scan(source, UnifaceDetector(cfg.detect), build_tracker(cfg.track), meta)

    frame_area = float(result.meta.width * result.meta.height)
    baseline_index = index_from_detections(result.raw, result.meta.size, cfg.heal)
    healed_index = result.timeline.heal(result.meta.n_frames, result.meta.size, cfg.heal)

    return BenchReport(
        source=source,
        n_frames=result.meta.n_frames,
        n_tracks=result.n_tracks,
        baseline=_evaluate("per-frame", baseline_index, annotations, frame_area, threshold),
        healed=_evaluate("tracked", healed_index, annotations, frame_area, threshold),
    )
