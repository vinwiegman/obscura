"""Selective redaction, end to end on a synthetic two-person clip.

Two people cross the frame in opposite directions, and each is split into two
tracks by a scripted detector dropout. The test asserts what a reviewer would
actually check: after picking one person to keep visible, that person's face is
still legible in the output and the other person's is not.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from obscura.config import HealConfig, IdentityConfig, RedactStyle, RunConfig, TrackConfig
from obscura.geometry import Box
from obscura.identity import TrackSampler, tracks_to_redact
from obscura.pipeline import render_scan, review, scan
from obscura.track import IouTracker
from obscura.types import Detection

N_FRAMES = 48
SIZE = (400, 240)
GAP = range(20, 26)
"""Frames where both faces are missed, splitting each person into two tracks."""

TRACK_BUFFER = 3
"""Shorter than GAP, so the tracker really does retire and reissue ids."""

ALICE_SHADE = 255
BOB_SHADE = 120
"""The two faces have to look different or no recognizer could separate them."""


def alice(i: int) -> Box:
    x = 20 + 5 * i
    return Box(x, 40, x + 56, 96)


def bob(i: int) -> Box:
    x = 320 - 5 * i
    return Box(x, 140, x + 56, 196)


def landmarks_for(box: Box) -> np.ndarray:
    cx, cy = box.center
    return np.array(
        [[cx - 14, cy - 10], [cx + 14, cy - 10], [cx, cy], [cx - 10, cy + 14], [cx + 10, cy + 14]],
        dtype=np.float32,
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("selective") / "two.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, SIZE)
    if not writer.isOpened():
        pytest.skip("No MJPG encoder available in this OpenCV build")

    for i in range(N_FRAMES):
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        for who, shade in ((alice, ALICE_SHADE), (bob, BOB_SHADE)):
            x1, y1, x2, y2 = who(i).to_int()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (shade,) * 3, thickness=-1)
        writer.write(frame)
    writer.release()
    return path


class TwoPersonDetector:
    def __init__(self) -> None:
        self.i = -1

    def detect(self, frame):
        self.i += 1
        if self.i in GAP:
            return []
        return [
            Detection(box=who(self.i), score=0.95, landmarks=landmarks_for(who(self.i)))
            for who in (alice, bob)
        ]


class FakeRecognizer:
    """Embeds a face by how bright it is.

    Stands in for ArcFace so the test runs without weights, while still driving
    the real alignment, embedding and clustering code.
    """

    def get_embedding(self, image, landmarks=None):
        brightness = float(image.mean()) / 255.0
        vector = np.array([brightness, 1.0 - brightness], dtype=np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def reviewed(clip):
    cfg = RunConfig(
        heal=HealConfig(),
        identity=IdentityConfig(enabled=True, threshold=0.9, min_face=20),
        style=RedactStyle(method="fill", shape="rect", feather=0.0),
        fourcc="MJPG",
    )
    sampler = TrackSampler(cfg.identity)
    result = scan(
        clip,
        TwoPersonDetector(),
        IouTracker(TrackConfig(track_buffer=TRACK_BUFFER)),
        sampler=sampler,
    )
    people, warnings = review(result, cfg, sampler, recognizer=FakeRecognizer())
    return cfg, result, people, warnings


def topmost(people, result):
    """The person whose face sits highest in frame -- Alice, by construction."""

    def height(person):
        return min(
            box.y1
            for track_id in person.track_ids
            for box in result.timeline.observations(track_id).values()
        )

    return min(people, key=height)


def test_the_dropout_really_does_split_each_person(reviewed):
    """Without clustering there would be four entries for two people."""
    _cfg, result, _people, _warnings = reviewed
    assert result.n_tracks == 4


def test_clustering_recovers_two_people_from_four_tracks(reviewed):
    _cfg, _result, people, _warnings = reviewed

    assert len(people) == 2
    assert sorted(len(person.track_ids) for person in people) == [2, 2]


def test_every_person_carries_a_thumbnail(reviewed):
    _cfg, _result, people, _warnings = reviewed
    assert all(person.thumbnail for person in people)


def test_nobody_is_pinned_when_every_face_is_identifiable(reviewed):
    _cfg, _result, people, warnings = reviewed
    assert not any(person.always_redact for person in people)
    assert warnings == []


def brightness(frame: np.ndarray, box: Box) -> float:
    x1, y1, x2, y2 = box.clip(*SIZE).to_int()
    return float(frame[y1 + 6 : y2 - 6, x1 + 6 : x2 - 6].mean())


def test_keeping_one_person_visible_blurs_only_the_other(reviewed, tmp_path, clip):
    """The claim the feature rests on, checked in pixels."""
    cfg, result, people, _warnings = reviewed
    redact = tracks_to_redact(people, selected={topmost(people, result).id}, policy="except")

    destination = tmp_path / "selective.avi"
    render_scan(clip, destination, result, cfg, only=redact)

    capture = cv2.VideoCapture(str(destination))
    try:
        for i in range(N_FRAMES):
            ok, frame = capture.read()
            assert ok, f"output truncated at frame {i}"
            assert brightness(frame, alice(i)) > 200, f"kept face was blurred at frame {i}"
            assert brightness(frame, bob(i)) < 60, f"blurred face survived at frame {i}"
    finally:
        capture.release()


def test_selecting_nobody_blurs_both_people(reviewed, tmp_path, clip):
    cfg, result, people, _warnings = reviewed
    redact = tracks_to_redact(people, selected=set(), policy="except")

    destination = tmp_path / "all.avi"
    render_scan(clip, destination, result, cfg, only=redact)

    capture = cv2.VideoCapture(str(destination))
    try:
        for i in range(N_FRAMES):
            ok, frame = capture.read()
            assert ok
            assert brightness(frame, alice(i)) < 60
            assert brightness(frame, bob(i)) < 60
    finally:
        capture.release()


def test_the_gap_frames_are_still_covered_for_the_blurred_person(reviewed, tmp_path, clip):
    """Selective redaction must not cost the healing that stops leaks."""
    cfg, result, people, _warnings = reviewed
    redact = tracks_to_redact(people, selected={topmost(people, result).id}, policy="except")

    destination = tmp_path / "gap.avi"
    render_scan(clip, destination, result, cfg, only=redact)

    capture = cv2.VideoCapture(str(destination))
    try:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    for i in GAP:
        assert brightness(frames[i], bob(i)) < 60, f"leak during detector gap at frame {i}"
