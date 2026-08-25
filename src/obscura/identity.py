"""Grouping tracks into people, so a reviewer can choose who gets blurred.

A track is one unbroken sighting, not a person. Someone who turns away, is
occluded, or walks out and back in produces a fresh track each time -- ten
tracks for one person on ordinary footage is unremarkable. Selecting at track
level therefore means a gallery full of duplicates where missing one entry
leaves that person exposed for part of the video.

So each track is embedded with a face recognition model and tracks that match
are merged. What the reviewer sees is people, not sightings.

The safety default runs the other way from the rest of this tool. Blur-everyone
fails safe: a mistake costs an over-blurred patch of wall. Blurring only *some*
people fails open, because a track that cannot be matched confidently is a real
person left identifiable. Anything too small, too low quality, or missing
landmarks is therefore pinned to always-redact and cannot be deselected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import cv2
import numpy as np

from .config import IdentityConfig
from .geometry import Box, iou
from .types import Detection

Policy = Literal["except", "only"]


class Recognizer(Protocol):
    def get_embedding(
        self, image: np.ndarray, landmarks: np.ndarray | None = ...
    ) -> np.ndarray: ...


@dataclass(slots=True)
class Person:
    """One identity in the footage, backed by one or more tracks."""

    id: int
    track_ids: list[int]
    n_frames: int
    first_frame: int
    last_frame: int
    thumbnail: bytes | None = None
    """JPEG bytes for the review gallery."""

    always_redact: bool = False
    reason: str | None = None
    """Why this person cannot be deselected, when ``always_redact`` is set."""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "tracks": len(self.track_ids),
            "frames": self.n_frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "always_redact": self.always_redact,
            "reason": self.reason,
            "has_thumbnail": self.thumbnail is not None,
        }


@dataclass(slots=True)
class _Sample:
    """The best-looking frame seen for one track."""

    track_id: int
    frame_index: int
    area: float
    thumbnail: np.ndarray
    aligned: np.ndarray | None = None
    embedding: np.ndarray | None = None


class TrackSampler:
    """Retains one representative crop per track during the scan pass.

    Only the current best crop per track is held, so memory is bounded by the
    number of tracks rather than the length of the video.
    """

    def __init__(self, cfg: IdentityConfig) -> None:
        self._cfg = cfg
        self._samples: dict[int, _Sample] = {}

    def observe(
        self,
        frame: np.ndarray,
        frame_index: int,
        track_id: int,
        box: Box,
        detections: list[Detection],
    ) -> None:
        """Offer one track's box in one frame as a candidate representative."""
        height, width = frame.shape[:2]
        clipped = box.clip(width, height)
        area = clipped.area
        if area <= 0:
            return

        current = self._samples.get(track_id)
        # Bigger is better: more pixels on the face means a more reliable
        # embedding and a thumbnail a human can actually recognise.
        if current is not None and area <= current.area:
            return

        x1, y1, x2, y2 = clipped.to_int()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        detection = _matching_detection(box, detections)
        self._samples[track_id] = _Sample(
            track_id=track_id,
            frame_index=frame_index,
            area=area,
            thumbnail=_fit(crop, self._cfg.thumb_size),
            aligned=_align(frame, detection, crop),
        )

    def samples(self) -> dict[int, _Sample]:
        return self._samples


def _matching_detection(box: Box, detections: list[Detection]) -> Detection | None:
    """Find the detection behind a track box.

    Tracker output is Kalman-smoothed and so never exactly equals a detection;
    the landmarks have to be recovered by overlap.
    """
    best, best_score = None, 0.0
    for detection in detections:
        score = iou(box, detection.box)
        if score > best_score:
            best, best_score = detection, score
    return best if best_score > 0.3 else None


def _align(frame: np.ndarray, detection: Detection | None, crop: np.ndarray) -> np.ndarray | None:
    """Warp the face onto canonical landmark positions for the recognizer.

    Alignment is what makes embeddings comparable across head poses. Without
    landmarks there is nothing to align to, and the face stays unmatched.
    """
    if detection is None or not detection.has_landmarks:
        return None
    try:
        from uniface.face_utils import face_alignment
    except ImportError:
        # No uniface means no recognizer either, so this path is only reached by
        # tests driving a stub. A plain resize keeps them runnable offline.
        return cv2.resize(crop, (112, 112))
    aligned, _matrix = face_alignment(frame, detection.landmarks)
    return aligned


def _fit(crop: np.ndarray, max_side: int) -> np.ndarray:
    height, width = crop.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return crop.copy()
    return cv2.resize(crop, (max(1, int(width * scale)), max(1, int(height * scale))))


def build_recognizer(cfg: IdentityConfig):
    """Construct the uniface recognition model, or None if unavailable."""
    from importlib import import_module

    try:
        module = import_module("uniface.recognition")
    except ImportError:
        return None
    factory = getattr(module, RECOGNIZERS.get(cfg.model.lower(), "ArcFace"), None)
    if factory is None:
        return None
    return factory(providers=cfg.providers) if cfg.providers else factory()


RECOGNIZERS = {
    "arcface": "ArcFace",
    "adaface": "AdaFace",
    "edgeface": "EdgeFace",
    "mobileface": "MobileFace",
    "sphereface": "SphereFace",
}


def embed(samples: dict[int, _Sample], recognizer: Recognizer | None) -> None:
    """Attach a unit-norm embedding to every sample that can carry one."""
    if recognizer is None:
        return
    for sample in samples.values():
        if sample.aligned is None:
            continue
        vector = np.asarray(recognizer.get_embedding(sample.aligned), dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        # A zero vector would make every cosine similarity NaN and silently
        # collapse the clustering.
        if norm > 1e-6:
            sample.embedding = vector / norm


def cluster(embeddings: dict[int, np.ndarray], threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering over cosine similarity.

    Average linkage rather than single linkage: with single linkage one
    ambiguous track bridges two people into a single cluster, and merging two
    people is exactly the failure that un-blurs someone.
    """
    track_ids = sorted(embeddings)
    if not track_ids:
        return []

    matrix = np.stack([embeddings[t] for t in track_ids])
    similarity = matrix @ matrix.T  # unit-norm vectors, so this is cosine
    clusters = [[i] for i in range(len(track_ids))]

    while len(clusters) > 1:
        best_pair, best_score = None, threshold
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                score = float(similarity[np.ix_(clusters[a], clusters[b])].mean())
                if score >= best_score:
                    best_pair, best_score = (a, b), score
        if best_pair is None:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)

    return [sorted(track_ids[i] for i in group) for group in clusters]


def build_people(
    timeline_spans: dict[int, tuple[int, int, int]],
    samples: dict[int, _Sample],
    cfg: IdentityConfig,
) -> list[Person]:
    """Group tracks into people.

    ``timeline_spans`` maps a track id to ``(n_frames, first_frame, last_frame)``.
    """
    reliable: dict[int, np.ndarray] = {}
    unreliable: dict[int, str] = {}

    for track_id in timeline_spans:
        sample = samples.get(track_id)
        if sample is None:
            unreliable[track_id] = "no usable frame captured"
        elif min(sample.thumbnail.shape[:2]) < cfg.min_face:
            unreliable[track_id] = f"face too small to identify (<{cfg.min_face}px)"
        elif sample.embedding is None:
            unreliable[track_id] = (
                "recognition model unavailable"
                if sample.aligned is not None
                else "no landmarks, so the face could not be matched"
            )
        else:
            reliable[track_id] = sample.embedding

    groups = cluster(reliable, cfg.threshold)
    groups.extend([track_id] for track_id in sorted(unreliable))

    people: list[Person] = []
    for group in groups:
        spans = [timeline_spans[t] for t in group]
        reason = next((unreliable[t] for t in group if t in unreliable), None)
        best = max(
            (samples[t] for t in group if t in samples),
            key=lambda s: s.area,
            default=None,
        )
        people.append(
            Person(
                id=0,  # assigned below, after ordering
                track_ids=list(group),
                n_frames=sum(span[0] for span in spans),
                first_frame=min(span[1] for span in spans),
                last_frame=max(span[2] for span in spans),
                thumbnail=_encode(best.thumbnail) if best is not None else None,
                always_redact=reason is not None,
                reason=reason,
            )
        )

    # Order by first appearance so ids are stable and read naturally in the UI.
    people.sort(key=lambda p: (p.first_frame, p.track_ids[0]))
    for index, person in enumerate(people, start=1):
        person.id = index
    return people


def _encode(image: np.ndarray) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buffer.tobytes() if ok else None


def tracks_to_redact(people: list[Person], selected: set[int], policy: Policy) -> set[int]:
    """Resolve a reviewer's selection into the set of track ids to blur.

    ``policy="except"`` blurs everyone the reviewer did not pick; ``"only"``
    blurs just those they did. Always-redact people are added under both, which
    is what stops an unrecognised face from slipping through.
    """
    redact: set[int] = set()
    for person in people:
        chosen = person.id in selected
        blur = not chosen if policy == "except" else chosen
        if blur or person.always_redact:
            redact.update(person.track_ids)
    return redact


@dataclass(slots=True)
class ReviewSelection:
    policy: Policy = "except"
    selected: set[int] = field(default_factory=set)
