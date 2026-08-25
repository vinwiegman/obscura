"""End-to-end check on a synthetic video with a detector that misses frames.

This is the claim the project rests on: a detector that drops frames leaks faces,
and healing the track timeline recovers them. Here the ground truth is known
exactly, so the leak can be counted rather than estimated.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from obscura.bench import coverage
from obscura.config import HealConfig, RedactStyle, RunConfig, TrackConfig
from obscura.geometry import Box
from obscura.pipeline import build_index, process, scan
from obscura.timeline import index_from_detections
from obscura.track import EkfTracker, IouTracker
from obscura.types import Detection

N_FRAMES = 60
SIZE = (320, 240)
DROPPED = {12, 13, 14, 15, 30, 44, 45}
"""Frames where the fake detector "fails" -- motion blur, a profile turn, occlusion."""


def true_box(frame_index: int) -> Box:
    """A face tracking steadily left to right."""
    x = 20 + 4 * frame_index
    return Box(x, 90, x + 50, 150)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, SIZE)
    if not writer.isOpened():
        pytest.skip("No MJPG encoder available in this OpenCV build")

    rng = np.random.default_rng(1)
    for frame_index in range(N_FRAMES):
        frame = rng.integers(0, 256, size=(SIZE[1], SIZE[0], 3), dtype=np.uint8)
        x1, y1, x2, y2 = true_box(frame_index).to_int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
        writer.write(frame)
    writer.release()
    return path


class FlakyDetector:
    """Returns the true box, except on the frames in ``DROPPED``."""

    def __init__(self) -> None:
        self.frame_index = -1

    def detect(self, frame):
        self.frame_index += 1
        if self.frame_index in DROPPED:
            return []
        box = true_box(self.frame_index)
        cx, cy = box.center
        landmarks = np.array(
            [[cx - 12, cy - 8], [cx + 12, cy - 8], [cx, cy], [cx - 9, cy + 12], [cx + 9, cy + 12]],
            dtype=np.float32,
        )
        return [Detection(box=box, score=0.95, landmarks=landmarks)]


class TailDropDetector:
    """Loses the face permanently for the final ten frames."""

    def __init__(self) -> None:
        self.frame_index = -1

    def detect(self, frame):
        self.frame_index += 1
        if self.frame_index >= N_FRAMES - 10:
            return []
        return [Detection(box=true_box(self.frame_index), score=0.95)]


@pytest.fixture
def scanned(clip):
    return scan(clip, FlakyDetector(), IouTracker(TrackConfig(track_buffer=30)))


def test_scan_reads_every_frame(scanned):
    assert scanned.meta.n_frames == N_FRAMES
    assert len(scanned.raw) == N_FRAMES


def test_scan_finds_a_single_track(scanned):
    assert scanned.n_tracks == 1


def test_baseline_leaks_exactly_the_dropped_frames(scanned):
    cfg = HealConfig(margin=0.0, top_extra=0.0)
    index = index_from_detections(scanned.raw, scanned.meta.size, cfg)

    leaked = {i for i in range(N_FRAMES) if coverage(true_box(i), index[i]) < 0.9}

    assert leaked == DROPPED


def test_healing_covers_every_frame(scanned):
    index = scanned.timeline.heal(N_FRAMES, scanned.meta.size, HealConfig())

    leaked = [i for i in range(N_FRAMES) if coverage(true_box(i), index[i]) < 0.9]

    assert leaked == []


def test_healed_boxes_track_the_face_rather_than_freezing(scanned):
    """Interpolation must follow the motion; holding the last box would drift off."""
    index = scanned.timeline.heal(N_FRAMES, scanned.meta.size, HealConfig())

    middle = index[14][0]
    expected = true_box(14)

    assert abs(middle.center[0] - expected.center[0]) < 5


def test_ekf_keeps_a_moving_face_covered_after_detection_stops(clip):
    track_cfg = TrackConfig(track_buffer=30, ekf_max_misses=10)
    tracker = EkfTracker(IouTracker(track_cfg), track_cfg)
    result = scan(clip, TailDropDetector(), tracker)

    # No frozen trail fallback: coverage in the tail must come from EKF motion.
    index = result.timeline.heal(
        N_FRAMES,
        result.meta.size,
        HealConfig(trail=0, margin=0.18, top_extra=0.0),
    )

    assert all(index[frame] for frame in range(N_FRAMES - 10, N_FRAMES))
    assert index[-1][0].center[0] > index[N_FRAMES - 11][0].center[0]
    assert coverage(true_box(N_FRAMES - 1), index[-1]) >= 0.9


def test_process_writes_a_playable_video(clip, tmp_path):
    destination = tmp_path / "out.avi"
    cfg = RunConfig(
        heal=HealConfig(),
        style=RedactStyle(method="fill", shape="rect", feather=0.0),
        fourcc="MJPG",
    )

    report = process(
        clip, destination, cfg, detector=FlakyDetector(), tracker=IouTracker(TrackConfig())
    )

    assert destination.exists()
    assert report.meta.n_frames == N_FRAMES
    # Healing adds coverage the detector never produced.
    assert report.n_redactions > report.n_detections

    capture = cv2.VideoCapture(str(destination))
    try:
        for frame_index in range(N_FRAMES):
            ok, frame = capture.read()
            assert ok, f"output truncated at frame {frame_index}"
            x1, y1, x2, y2 = true_box(frame_index).clip(*SIZE).to_int()
            patch = frame[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5]
            # The face was solid white; every frame of the output must be dark there.
            assert patch.mean() < 60, f"face still visible at frame {frame_index}"
    finally:
        capture.release()


def _run(clip, tmp_path, monkeypatch, **overrides):
    from obscura import video as video_module

    for name, value in overrides.pop("patches", {}).items():
        monkeypatch.setattr(video_module, name, value)
    cfg = RunConfig(style=RedactStyle(method="fill"), fourcc="MJPG", **overrides)
    return process(
        clip, tmp_path / "out.avi", cfg, detector=FlakyDetector(), tracker=IouTracker(TrackConfig())
    )


def test_keep_audio_without_ffmpeg_warns_instead_of_failing_quietly(clip, tmp_path, monkeypatch):
    """Asking for audio and silently getting none is the worst possible outcome."""
    report = _run(
        clip, tmp_path, monkeypatch, keep_audio=True, patches={"has_ffmpeg": lambda: False}
    )

    assert report.output.exists()
    assert any("ffmpeg" in w for w in report.warnings)


def test_dropped_audio_is_reported_when_the_source_has_a_track(clip, tmp_path, monkeypatch):
    report = _run(clip, tmp_path, monkeypatch, patches={"probe_has_audio": lambda _: True})

    assert any("--keep-audio" in w for w in report.warnings)


def test_silent_source_produces_no_audio_warning(clip, tmp_path, monkeypatch):
    report = _run(clip, tmp_path, monkeypatch, patches={"probe_has_audio": lambda _: False})

    assert report.warnings == []


def test_failed_remux_keeps_the_video_and_warns(clip, tmp_path, monkeypatch):
    """A remux that fails must still leave a usable, silent video behind."""
    report = _run(
        clip,
        tmp_path,
        monkeypatch,
        keep_audio=True,
        patches={"has_ffmpeg": lambda: True, "remux_audio": lambda *a: False},
    )

    assert report.output.exists()
    assert not report.output.with_name("out.silent.avi").exists()
    assert any("could not copy the audio" in w for w in report.warnings)


def test_single_pass_config_skips_healing(scanned):
    cfg = RunConfig(single_pass=True, heal=HealConfig(margin=0.0, top_extra=0.0))

    index = build_index(scanned, cfg)

    assert all(not index[i] for i in DROPPED)
