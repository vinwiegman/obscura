"""Multi-object tracking, adapted from uniface's BYTETracker.

A pure-Python IoU tracker stands in when uniface is unavailable, which keeps the
pipeline testable offline. It is weaker through occlusion -- BYTETracker's Kalman
prediction and two-stage association recover tracks the IoU version drops -- so
the fallback is a convenience, not a supported mode.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import TrackConfig
from .geometry import Box, iou


class Tracker(Protocol):
    def update(self, detections: list[tuple[Box, float]]) -> list[tuple[int, Box]]: ...


def build(cfg: TrackConfig) -> Tracker:
    try:
        from uniface.tracking import BYTETracker
    except ImportError:
        return IouTracker(cfg)
    return ByteTrackAdapter(BYTETracker, cfg)


class ByteTrackAdapter:
    """Feeds ``(N, 5)`` detection arrays in, reads ``(M, 5)`` track arrays out."""

    def __init__(self, factory, cfg: TrackConfig) -> None:
        self._tracker = factory(
            track_thresh=cfg.track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
        )

    def update(self, detections: list[tuple[Box, float]]) -> list[tuple[int, Box]]:
        if detections:
            array = np.array(
                [[b.x1, b.y1, b.x2, b.y2, score] for b, score in detections],
                dtype=np.float32,
            )
        else:
            # The tracker still has to run on empty frames: that is how it ages
            # out lost tracks and keeps its Kalman states moving.
            array = np.empty((0, 5), dtype=np.float32)

        tracks = np.asarray(self._tracker.update(array))
        if tracks.size == 0:
            return []

        results = []
        for row in tracks.reshape(-1, tracks.shape[-1]):
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            results.append((int(row[4]), Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))
        return results


class IouTracker:
    """Greedy IoU association with a lost-track buffer."""

    def __init__(self, cfg: TrackConfig) -> None:
        self._cfg = cfg
        self._next_id = 1
        self._tracks: dict[int, tuple[Box, int]] = {}  # id -> (box, frames since seen)

    def update(self, detections: list[tuple[Box, float]]) -> list[tuple[int, Box]]:
        kept = [(box, score) for box, score in detections if score >= self._cfg.track_thresh]
        unmatched = set(range(len(kept)))
        matched: dict[int, Box] = {}

        pairs = sorted(
            (
                (iou(track_box, kept[i][0]), track_id, i)
                for track_id, (track_box, _) in self._tracks.items()
                for i in range(len(kept))
            ),
            reverse=True,
        )

        used_tracks: set[int] = set()
        for score, track_id, i in pairs:
            if score < 1.0 - self._cfg.match_thresh:
                break
            if track_id in used_tracks or i not in unmatched:
                continue
            used_tracks.add(track_id)
            unmatched.discard(i)
            matched[track_id] = kept[i][0]

        for i in unmatched:
            matched[self._next_id] = kept[i][0]
            self._next_id += 1

        # Age unmatched tracks; drop them once past the buffer.
        aged: dict[int, tuple[Box, int]] = {}
        for track_id, (box, missed) in self._tracks.items():
            if track_id not in matched and missed + 1 <= self._cfg.track_buffer:
                aged[track_id] = (box, missed + 1)
        for track_id, box in matched.items():
            aged[track_id] = (box, 0)
        self._tracks = aged

        return sorted(matched.items())
