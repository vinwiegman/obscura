"""Multi-object tracking, adapted from uniface's BYTETracker.

A pure-Python IoU tracker stands in when uniface is unavailable, which keeps the
pipeline testable offline. It is weaker through occlusion -- BYTETracker's Kalman
prediction and two-stage association recover tracks the IoU version drops -- so
the fallback is a convenience, not a supported mode.

BYTETracker does not emit a box for every temporarily lost track. ``EkfTracker``
adds one nonlinear bounding-box filter per identity and emits bounded predictions
for those missing frames. The timeline marks them as predictions so its two-pass
healer can still replace them with interpolation when a later real observation
is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
        base: Tracker = IouTracker(cfg)
    else:
        base = ByteTrackAdapter(BYTETracker, cfg)
    return EkfTracker(base, cfg) if cfg.ekf_max_misses > 0 else base


@dataclass(slots=True)
class _EkfState:
    filter: BoxEkf
    missed: int = 0


class BoxEkf:
    """Extended Kalman filter for one face box.

    The state is ``(cx, cy, log(w), log(h), vx, vy, v_log_w, v_log_h)``.
    Width and height are represented in log space so predictions can never turn
    negative. Converting those log sizes back to pixel measurements is nonlinear,
    which is the EKF observation model.
    """

    _MIN_SIZE = 1.0

    def __init__(self, box: Box) -> None:
        width = max(box.width, self._MIN_SIZE)
        height = max(box.height, self._MIN_SIZE)
        cx, cy = box.center
        self.x = np.array(
            [cx, cy, math.log(width), math.log(height), 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        scale = max(width, height)
        pos_var = max(1.0, 0.08 * scale) ** 2
        vel_var = max(1.0, 0.25 * scale) ** 2
        self.p = np.diag([pos_var, pos_var, 0.04, 0.04, vel_var, vel_var, 0.04, 0.04])

        self.f = np.eye(8, dtype=np.float64)
        self.f[0, 4] = self.f[1, 5] = self.f[2, 6] = self.f[3, 7] = 1.0

        pos_q = max(0.5, 0.015 * scale) ** 2
        vel_q = max(0.25, 0.0075 * scale) ** 2
        self.q = np.diag([pos_q, pos_q, 0.0025, 0.0025, vel_q, vel_q, 0.001, 0.001])

    def predict(self, velocity_decay: float = 1.0) -> Box:
        width, height = self._sizes()
        max_speed = 0.35 * max(width, height)
        speed = float(np.linalg.norm(self.x[4:6]))
        if speed > max_speed > 0:
            self.x[4:6] *= max_speed / speed
        self.x[6:8] = np.clip(self.x[6:8], math.log(0.85), math.log(1.15))

        self.x = self.f @ self.x
        self.x[4:8] *= min(max(velocity_decay, 0.0), 1.0)
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.box

    def update(self, box: Box) -> Box:
        width = max(box.width, self._MIN_SIZE)
        height = max(box.height, self._MIN_SIZE)
        cx, cy = box.center
        measurement = np.array([cx, cy, width, height], dtype=np.float64)

        predicted_width, predicted_height = self._sizes()
        expected = np.array(
            [self.x[0], self.x[1], predicted_width, predicted_height], dtype=np.float64
        )
        h = np.zeros((4, 8), dtype=np.float64)
        h[0, 0] = h[1, 1] = 1.0
        h[2, 2] = predicted_width
        h[3, 3] = predicted_height

        pos_sigma = max(1.0, 0.05 * max(width, height))
        r = np.diag(
            [
                pos_sigma**2,
                pos_sigma**2,
                max(1.0, 0.08 * width) ** 2,
                max(1.0, 0.08 * height) ** 2,
            ]
        )
        innovation = measurement - expected
        innovation_cov = h @ self.p @ h.T + r
        cross_cov = self.p @ h.T
        try:
            gain = np.linalg.solve(innovation_cov, cross_cov.T).T
        except np.linalg.LinAlgError:
            gain = cross_cov @ np.linalg.pinv(innovation_cov)

        self.x = self.x + gain @ innovation
        # Joseph form keeps the covariance symmetric and positive semidefinite.
        identity = np.eye(8, dtype=np.float64)
        residual = identity - gain @ h
        self.p = residual @ self.p @ residual.T + gain @ r @ gain.T
        self.p = (self.p + self.p.T) * 0.5
        return self.box

    @property
    def box(self) -> Box:
        width, height = self._sizes()
        cx, cy = float(self.x[0]), float(self.x[1])
        return Box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    def _sizes(self) -> tuple[float, float]:
        # The clamp is only a numerical guard for corrupt/extreme input. It is
        # far outside the useful range of face sizes in real video.
        width = math.exp(float(np.clip(self.x[2], -6.0, 14.0)))
        height = math.exp(float(np.clip(self.x[3], -6.0, 14.0)))
        return max(width, self._MIN_SIZE), max(height, self._MIN_SIZE)


class EkfTracker:
    """Add per-track box prediction to a detector-associated base tracker."""

    def __init__(self, tracker: Tracker, cfg: TrackConfig) -> None:
        self._tracker = tracker
        self._cfg = cfg
        self._states: dict[int, _EkfState] = {}
        self._aliases: dict[int, int] = {}
        self._next_id = 1
        self._predicted_ids: frozenset[int] = frozenset()

    @property
    def predicted_ids(self) -> frozenset[int]:
        """Track ids whose boxes in the latest result are EKF predictions."""
        return self._predicted_ids

    def update(self, detections: list[tuple[Box, float]]) -> list[tuple[int, Box]]:
        base_observations = self._tracker.update(detections)
        predictions = {
            track_id: state.filter.predict(velocity_decay=0.97)
            for track_id, state in self._states.items()
        }

        results: dict[int, Box] = {}
        observed_ids: set[int] = set()
        for base_id, box in base_observations:
            track_id = self._aliases.get(base_id)
            if track_id not in self._states or track_id in observed_ids:
                track_id = self._match_reacquired(box, predictions, observed_ids)
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
                self._aliases[base_id] = track_id

            state = self._states.get(track_id)
            if state is None:
                state = _EkfState(BoxEkf(box))
                self._states[track_id] = state
            else:
                state.filter.update(box)
            state.missed = 0
            observed_ids.add(track_id)
            # A real detector box is safer than a smoothed box on observed
            # frames: filtering must never shrink coverage away from evidence.
            results[track_id] = box

        predicted_ids: set[int] = set()
        retire_after = max(self._cfg.track_buffer, self._cfg.ekf_max_misses)
        for track_id, state in list(self._states.items()):
            if track_id in observed_ids:
                continue
            state.missed += 1
            if state.missed <= self._cfg.ekf_max_misses:
                uncertainty_margin = min(0.25, state.missed * 0.01)
                results[track_id] = predictions[track_id].expand(uncertainty_margin)
                predicted_ids.add(track_id)
            if state.missed > retire_after:
                del self._states[track_id]
                self._aliases = {
                    base_id: canonical_id
                    for base_id, canonical_id in self._aliases.items()
                    if canonical_id != track_id
                }

        self._predicted_ids = frozenset(predicted_ids)
        return sorted(results.items())

    def _match_reacquired(
        self,
        observation: Box,
        predictions: dict[int, Box],
        unavailable: set[int],
    ) -> int | None:
        """Relink a new base-tracker id to a plausible recently lost face.

        BYTETracker can assign a fresh id after an occlusion. Without this
        handoff the old EKF becomes a ghost blur while the new id covers the
        actual face. The gate is intentionally strict so nearby people in a
        crowd are not casually merged.
        """
        best: tuple[float, int] | None = None
        for track_id, predicted in predictions.items():
            if track_id in unavailable:
                continue
            state = self._states[track_id]
            if state.missed > self._cfg.ekf_max_misses:
                continue

            width_ratio = observation.width / max(predicted.width, BoxEkf._MIN_SIZE)
            height_ratio = observation.height / max(predicted.height, BoxEkf._MIN_SIZE)
            if not (0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0):
                continue

            dx = observation.center[0] - predicted.center[0]
            dy = observation.center[1] - predicted.center[1]
            scale = max(
                math.hypot(observation.width, observation.height),
                math.hypot(predicted.width, predicted.height),
                1.0,
            )
            distance = math.hypot(dx, dy) / scale
            overlap = iou(observation, predicted)
            if overlap < 0.1 and distance > 0.75:
                continue

            score = overlap + 0.5 * (1.0 - min(distance, 1.0))
            if best is None or score > best[0]:
                best = (score, track_id)
        return None if best is None else best[1]


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
