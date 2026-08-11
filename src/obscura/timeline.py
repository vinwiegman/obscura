"""Track bookkeeping and gap healing.

This module is where the tool earns its keep. A detector run frame-by-frame will
miss faces on motion blur, profile angles and partial occlusion; every miss is a
frame that ships an identifiable face. Because pass 1 records the whole video
before anything is rendered, a miss at frame ``t`` can be repaired using
detections from *after* ``t`` -- which a streaming, single-pass tool cannot do.

Only geometry is retained, so a 20-minute video costs a few megabytes here.
"""

from __future__ import annotations

from collections import defaultdict

from .config import HealConfig
from .geometry import Box

# frame index -> boxes to redact in that frame
FrameIndex = list[list[Box]]


class TrackTimeline:
    """Sparse per-track observations, accumulated during the scan pass."""

    def __init__(self) -> None:
        self._tracks: dict[int, dict[int, Box]] = defaultdict(dict)

    def add(self, frame: int, track_id: int, box: Box) -> None:
        self._tracks[track_id][frame] = box

    @property
    def track_ids(self) -> list[int]:
        return sorted(self._tracks)

    def observations(self, track_id: int) -> dict[int, Box]:
        return self._tracks[track_id]

    def __len__(self) -> int:
        return len(self._tracks)

    def heal(self, n_frames: int, size: tuple[int, int], cfg: HealConfig) -> FrameIndex:
        """Expand sparse observations into dense per-frame coverage."""
        width, height = size
        index: FrameIndex = [[] for _ in range(n_frames)]

        for track_id in self.track_ids:
            for frame, box in _heal_track(self._tracks[track_id], n_frames, cfg).items():
                grown = box.expand(cfg.margin, cfg.top_extra).clip(width, height)
                if not grown.is_empty:
                    index[frame].append(grown)

        return index


def _heal_track(obs: dict[int, Box], n_frames: int, cfg: HealConfig) -> dict[int, Box]:
    """Fill gaps within one track, then extend past its first and last sighting."""
    if not obs:
        return {}

    frames = sorted(obs)
    dense: dict[int, Box] = dict(obs)

    # Bridge interior gaps by interpolating between the flanking observations.
    for start, end in zip(frames, frames[1:], strict=False):
        missing = end - start - 1
        if missing <= 0 or missing > cfg.max_gap:
            continue
        span = end - start
        for frame in range(start + 1, end):
            dense[frame] = Box.lerp(obs[start], obs[end], (frame - start) / span)

    # Extend outward. A face is rarely first detected on the frame it appears --
    # it is detected once it turns far enough toward the camera, by which point
    # earlier frames have already leaked it.
    first, last = frames[0], frames[-1]
    for frame in range(max(0, first - cfg.lead), first):
        dense.setdefault(frame, obs[first])
    for frame in range(last + 1, min(n_frames, last + cfg.trail + 1)):
        dense.setdefault(frame, obs[last])

    return {f: b for f, b in dense.items() if 0 <= f < n_frames}


def index_from_detections(
    detections: list[list[Box]], size: tuple[int, int], cfg: HealConfig
) -> FrameIndex:
    """Build a frame index from raw detections, with no healing.

    The naive baseline: dilation still applies, so benchmarks isolate the effect
    of tracking rather than measuring the margin twice.
    """
    width, height = size
    index: FrameIndex = []
    for frame_boxes in detections:
        kept = []
        for box in frame_boxes:
            grown = box.expand(cfg.margin, cfg.top_extra).clip(width, height)
            if not grown.is_empty:
                kept.append(grown)
        index.append(kept)
    return index
