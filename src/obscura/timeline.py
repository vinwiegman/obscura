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
        self._predictions: dict[int, dict[int, Box]] = defaultdict(dict)

    def add(self, frame: int, track_id: int, box: Box, *, predicted: bool = False) -> None:
        target = self._predictions if predicted else self._tracks
        target[track_id][frame] = box

    @property
    def track_ids(self) -> list[int]:
        return sorted(self._tracks.keys() | self._predictions.keys())

    def observations(self, track_id: int) -> dict[int, Box]:
        return self._tracks[track_id]

    def __len__(self) -> int:
        return len(self._tracks.keys() | self._predictions.keys())

    def spans(self) -> dict[int, tuple[int, int, int]]:
        """Per track: ``(frames seen, first frame, last frame)``.

        Prediction-only tracks are included. Identity grouping keys off this, and
        a track missing from it would belong to no person, so nothing could ever
        select it for redaction -- a silent way to leak a face.
        """
        spans: dict[int, tuple[int, int, int]] = {}
        for track_id in self.track_ids:
            frames = set(self._tracks.get(track_id, {})) | set(self._predictions.get(track_id, {}))
            if frames:
                spans[track_id] = (len(frames), min(frames), max(frames))
        return spans

    def heal(
        self,
        n_frames: int,
        size: tuple[int, int],
        cfg: HealConfig,
        only: set[int] | None = None,
    ) -> FrameIndex:
        """Expand sparse observations into dense per-frame coverage.

        ``only`` restricts the output to those track ids, which is how selective
        redaction is applied: everything else in the timeline is left visible.

        Stitching runs first, so a group can span several original track ids. A
        group is redacted if *any* of its ids was selected -- if motion
        stitching wrongly joined two people, over-blurring is the survivable
        error and leaving one visible is not.
        """
        width, height = size
        index: FrameIndex = [[] for _ in range(n_frames)]

        for track_ids, observations, predictions in _stitch_tracklets(
            self._tracks, self._predictions, cfg
        ):
            if only is not None and not (track_ids & only):
                continue
            dense = _heal_track(observations, predictions, n_frames, cfg)
            for frame, box in dense.items():
                grown = box.expand(cfg.margin, cfg.top_extra).clip(width, height)
                if not grown.is_empty:
                    index[frame].append(grown)

        return index


def _stitch_tracklets(
    tracks: dict[int, dict[int, Box]],
    predictions: dict[int, dict[int, Box]],
    cfg: HealConfig,
) -> list[tuple[set[int], dict[int, Box], dict[int, Box]]]:
    """Join plausible non-overlapping fragments before filling their gaps.

    A detector can reacquire the same face under a fresh tracker id. Keeping the
    fragments separate leaves the old EKF prediction drifting while the new id
    starts later. Since rendering waits for the whole scan, endpoint geometry on
    both sides can safely identify plausible handoffs and interpolation can then
    replace the forward-only predictions.

    Each group carries the original track ids it absorbed, so selective
    redaction can still tell which people a stitched group belongs to.
    """
    fragments = sorted(
        (
            (track_id, dict(observations), dict(predictions.get(track_id, {})))
            for track_id, observations in tracks.items()
            if observations
        ),
        key=lambda item: min(item[1]),
    )
    stitched: list[tuple[set[int], dict[int, Box], dict[int, Box]]] = []

    for track_id, observations, forecast in fragments:
        first = min(observations)
        first_box = observations[first]
        best: tuple[float, int] | None = None

        for index, (_candidate_ids, candidate_obs, _candidate_forecast) in enumerate(stitched):
            last = max(candidate_obs)
            missing = first - last - 1
            if missing < 0 or missing > cfg.max_gap:
                continue

            last_box = candidate_obs[last]
            width_ratio = first_box.width / max(last_box.width, 1.0)
            height_ratio = first_box.height / max(last_box.height, 1.0)
            if not (0.4 <= width_ratio <= 2.5 and 0.4 <= height_ratio <= 2.5):
                continue

            dx = first_box.center[0] - last_box.center[0]
            dy = first_box.center[1] - last_box.center[1]
            distance = (dx * dx + dy * dy) ** 0.5
            frame_span = first - last
            scale = max(last_box.width, last_box.height, first_box.width, first_box.height, 1.0)
            speed = distance / max(frame_span, 1)
            if speed > 0.25 * scale:
                continue

            score = distance / scale + missing / max(cfg.max_gap, 1) * 0.25
            if best is None or score < best[0]:
                best = (score, index)

        if best is None:
            stitched.append(({track_id}, observations, forecast))
            continue

        candidate_ids, candidate_obs, candidate_forecast = stitched[best[1]]
        candidate_ids.add(track_id)
        candidate_obs.update(observations)
        candidate_forecast.update(forecast)

    # Prediction-only timelines are unusual but valid for callers constructing
    # TrackTimeline directly, so preserve them rather than silently dropping data.
    for track_id, forecast in predictions.items():
        if track_id not in tracks and forecast:
            stitched.append(({track_id}, {}, dict(forecast)))

    return stitched


def _heal_track(
    obs: dict[int, Box], predictions: dict[int, Box], n_frames: int, cfg: HealConfig
) -> dict[int, Box]:
    """Fill gaps, preferring hindsight interpolation over forward predictions."""
    if not obs:
        return {f: b for f, b in predictions.items() if 0 <= f < n_frames}

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
        dense.setdefault(frame, predictions.get(frame, obs[last]))

    # EKF predictions cover detector loss without a future observation. They
    # are inserted last so interpolation from real boxes on both sides wins.
    for frame, box in predictions.items():
        if 0 <= frame < n_frames:
            dense.setdefault(frame, box)

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
