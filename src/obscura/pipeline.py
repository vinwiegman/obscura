"""Two-pass redaction pipeline.

Pass 1 (``scan``) decodes the video, detects and tracks faces, and keeps only box
geometry. Healing then repairs the track timeline with hindsight. Pass 2
(``render``) decodes again and writes the redacted output.

Two decode passes cost roughly 15% more wall time than a streaming design, and
buy the ability to fix a dropped detection using evidence from later frames.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from . import redact, video
from .config import RunConfig
from .detect import Detector, UnifaceDetector
from .geometry import Box
from .timeline import FrameIndex, TrackTimeline, index_from_detections
from .track import Tracker
from .track import build as build_tracker
from .video import VideoMeta

Progress = Callable[[str, int, int], None]
"""``(stage, current, total)``; ``total`` is 0 when unknown."""


def _noop(stage: str, current: int, total: int) -> None:
    pass


@dataclass(slots=True)
class ScanResult:
    meta: VideoMeta
    raw: list[list[Box]]
    """Per-frame detections, exactly as the detector produced them."""

    timeline: TrackTimeline
    n_tracks: int


@dataclass(slots=True)
class RunReport:
    source: Path
    output: Path
    meta: VideoMeta
    n_tracks: int
    n_detections: int
    n_redactions: int
    """Total boxes drawn, after healing. The gap against ``n_detections`` is the
    coverage that per-frame detection would have missed."""

    scan_seconds: float
    render_seconds: float
    warnings: list[str] = dc_field(default_factory=list)
    """Things that silently degraded the output, most often a dropped audio track."""

    @property
    def fps(self) -> float:
        total = self.scan_seconds + self.render_seconds
        return self.meta.n_frames / total if total > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        duration = self.meta.n_frames / self.meta.fps if self.meta.fps else 0.0
        total = self.scan_seconds + self.render_seconds
        return duration / total if total > 0 else 0.0


def scan(
    path: Path,
    detector: Detector,
    tracker: Tracker,
    meta: VideoMeta | None = None,
    progress: Progress = _noop,
) -> ScanResult:
    """Pass 1: detect and track, retaining geometry only."""
    meta = meta or video.probe(path)
    timeline = TrackTimeline()
    raw: list[list[Box]] = []

    for frame_index, frame in enumerate(video.frames(path)):
        detections = detector.detect(frame)
        raw.append([box for box, _ in detections])
        for track_id, box in tracker.update(detections):
            timeline.add(frame_index, track_id, box)
        progress("scan", frame_index + 1, meta.n_frames)

    # Trust the decoded count over the container's claim.
    meta.n_frames = len(raw)
    return ScanResult(meta=meta, raw=raw, timeline=timeline, n_tracks=len(timeline))


def render(
    path: Path,
    destination: Path,
    index: FrameIndex,
    meta: VideoMeta,
    cfg: RunConfig,
    progress: Progress = _noop,
) -> None:
    """Pass 2: re-decode and write the redacted video."""
    with video.writer(destination, meta, cfg.fourcc) as sink:
        for frame_index, frame in enumerate(video.frames(path)):
            boxes = index[frame_index] if frame_index < len(index) else []
            sink.write(redact.apply(frame, boxes, cfg.style))
            progress("render", frame_index + 1, meta.n_frames)


def build_index(result: ScanResult, cfg: RunConfig) -> FrameIndex:
    if cfg.single_pass:
        return index_from_detections(result.raw, result.meta.size, cfg.heal)
    return result.timeline.heal(result.meta.n_frames, result.meta.size, cfg.heal)


def process(
    source: Path,
    destination: Path,
    cfg: RunConfig,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    progress: Progress = _noop,
) -> RunReport:
    """Redact one video end to end."""
    meta = video.probe(source)
    detector = detector or UnifaceDetector(cfg.detect)
    tracker = tracker or build_tracker(cfg.track)

    started = time.perf_counter()
    result = scan(source, detector, tracker, meta, progress)
    scan_seconds = time.perf_counter() - started

    index = build_index(result, cfg)

    warnings: list[str] = []
    target = destination
    silent = destination
    remuxing = False

    if cfg.keep_audio:
        if video.has_ffmpeg():
            remuxing = True
            silent = destination.with_name(f"{destination.stem}.silent{destination.suffix}")
        else:
            # Asking for audio and silently getting none is the worst outcome:
            # the file looks finished and the loss is only noticed on playback.
            warnings.append("--keep-audio needs ffmpeg on PATH; the output has no audio track.")
    elif video.probe_has_audio(source):
        warnings.append("Source has an audio track; the output does not. Use --keep-audio.")

    started = time.perf_counter()
    render(source, silent, index, result.meta, cfg, progress)
    render_seconds = time.perf_counter() - started

    if remuxing:
        if video.remux_audio(source, silent, target):
            silent.unlink(missing_ok=True)
        else:
            silent.replace(target)
            warnings.append(
                f"ffmpeg could not copy the audio into {target.suffix}; output is silent. "
                "An .mp4 destination is the safest container for this."
            )

    return RunReport(
        source=source,
        output=target,
        meta=result.meta,
        n_tracks=result.n_tracks,
        n_detections=sum(len(boxes) for boxes in result.raw),
        n_redactions=sum(len(boxes) for boxes in index),
        scan_seconds=scan_seconds,
        render_seconds=render_seconds,
        warnings=warnings,
    )
