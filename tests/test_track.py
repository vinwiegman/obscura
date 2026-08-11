from obscura.config import TrackConfig
from obscura.geometry import Box
from obscura.track import IouTracker


def test_a_moving_box_keeps_its_id():
    tracker = IouTracker(TrackConfig())

    first = tracker.update([(Box(0, 0, 40, 40), 0.9)])
    second = tracker.update([(Box(6, 0, 46, 40), 0.9)])

    assert first[0][0] == second[0][0]


def test_a_second_face_gets_its_own_id():
    tracker = IouTracker(TrackConfig())

    tracker.update([(Box(0, 0, 40, 40), 0.9)])
    tracks = tracker.update([(Box(0, 0, 40, 40), 0.9), (Box(200, 0, 240, 40), 0.9)])

    assert len({track_id for track_id, _ in tracks}) == 2


def test_a_track_survives_a_missed_frame_within_the_buffer():
    tracker = IouTracker(TrackConfig(track_buffer=5))

    original = tracker.update([(Box(0, 0, 40, 40), 0.9)])[0][0]
    tracker.update([])  # detector miss
    recovered = tracker.update([(Box(2, 0, 42, 40), 0.9)])[0][0]

    assert recovered == original


def test_a_track_is_retired_once_past_the_buffer():
    tracker = IouTracker(TrackConfig(track_buffer=1))

    original = tracker.update([(Box(0, 0, 40, 40), 0.9)])[0][0]
    for _ in range(3):
        tracker.update([])
    revived = tracker.update([(Box(0, 0, 40, 40), 0.9)])[0][0]

    assert revived != original


def test_low_confidence_detections_are_ignored():
    tracker = IouTracker(TrackConfig(track_thresh=0.5))
    assert tracker.update([(Box(0, 0, 40, 40), 0.2)]) == []


def test_distant_boxes_are_not_associated():
    tracker = IouTracker(TrackConfig())

    original = tracker.update([(Box(0, 0, 40, 40), 0.9)])[0][0]
    moved = tracker.update([(Box(300, 300, 340, 340), 0.9)])[0][0]

    assert moved != original
