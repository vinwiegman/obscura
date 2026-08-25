from obscura.config import TrackConfig
from obscura.geometry import Box
from obscura.track import BoxEkf, EkfTracker, IouTracker
from obscura.types import Detection


def det(x1, y1, x2, y2, score=0.9):
    return Detection(box=Box(x1, y1, x2, y2), score=score)


class ScriptedTracker:
    def __init__(self, frames):
        self.frames = iter(frames)

    def update(self, detections):
        return next(self.frames)


def test_a_moving_box_keeps_its_id():
    tracker = IouTracker(TrackConfig())

    first = tracker.update([det(0, 0, 40, 40)])
    second = tracker.update([det(6, 0, 46, 40)])

    assert first[0][0] == second[0][0]


def test_a_second_face_gets_its_own_id():
    tracker = IouTracker(TrackConfig())

    tracker.update([det(0, 0, 40, 40)])
    tracks = tracker.update([det(0, 0, 40, 40), det(200, 0, 240, 40)])

    assert len({track_id for track_id, _ in tracks}) == 2


def test_a_track_survives_a_missed_frame_within_the_buffer():
    tracker = IouTracker(TrackConfig(track_buffer=5))

    original = tracker.update([det(0, 0, 40, 40)])[0][0]
    tracker.update([])  # detector miss
    recovered = tracker.update([det(2, 0, 42, 40)])[0][0]

    assert recovered == original


def test_a_track_is_retired_once_past_the_buffer():
    tracker = IouTracker(TrackConfig(track_buffer=1))

    original = tracker.update([det(0, 0, 40, 40)])[0][0]
    for _ in range(3):
        tracker.update([])
    revived = tracker.update([det(0, 0, 40, 40)])[0][0]

    assert revived != original


def test_low_confidence_detections_are_ignored():
    tracker = IouTracker(TrackConfig(track_thresh=0.5))
    assert tracker.update([det(0, 0, 40, 40, score=0.2)]) == []


def test_distant_boxes_are_not_associated():
    tracker = IouTracker(TrackConfig())

    original = tracker.update([det(0, 0, 40, 40)])[0][0]
    moved = tracker.update([det(300, 300, 340, 340)])[0][0]

    assert moved != original


def test_box_ekf_learns_motion_and_predicts_forward():
    ekf = BoxEkf(Box(0, 0, 40, 40))

    for offset in (5, 10, 15, 20):
        ekf.predict()
        ekf.update(Box(offset, 0, offset + 40, 40))

    last_center = ekf.box.center[0]
    predicted = ekf.predict()

    assert predicted.center[0] > last_center
    assert predicted.width > 0
    assert predicted.height > 0


def test_ekf_tracker_emits_bounded_predictions_for_missed_faces():
    cfg = TrackConfig(track_buffer=5, ekf_max_misses=2)
    tracker = EkfTracker(IouTracker(cfg), cfg)

    first = tracker.update([det(0, 0, 40, 40)])
    track_id = first[0][0]
    tracker.update([det(5, 0, 45, 40)])

    miss_one = tracker.update([])
    miss_two = tracker.update([])
    miss_three = tracker.update([])

    assert miss_one[0][0] == track_id
    assert miss_two[0][0] == track_id
    assert miss_one[0][1].center[0] > first[0][1].center[0]
    assert tracker.predicted_ids == frozenset()
    assert miss_three == []


def test_ekf_tracker_marks_only_predicted_results():
    cfg = TrackConfig(ekf_max_misses=2)
    tracker = EkfTracker(IouTracker(cfg), cfg)

    observed = tracker.update([det(0, 0, 40, 40)])
    assert tracker.predicted_ids == frozenset()

    predicted = tracker.update([])
    assert tracker.predicted_ids == {observed[0][0]}
    assert predicted[0][0] == observed[0][0]


def test_reacquired_face_with_new_base_id_reuses_the_original_ekf_track():
    cfg = TrackConfig(ekf_max_misses=5)
    base = ScriptedTracker(
        [
            [(10, Box(0, 0, 40, 40))],
            [(10, Box(5, 0, 45, 40))],
            [],
            [(99, Box(15, 0, 55, 40))],
        ]
    )
    tracker = EkfTracker(base, cfg)

    original_id = tracker.update([])[0][0]
    tracker.update([])
    predicted = tracker.update([])
    reacquired = tracker.update([])

    assert predicted[0][0] == original_id
    assert tracker.predicted_ids == frozenset()
    assert reacquired == [(original_id, Box(15, 0, 55, 40))]


def test_distant_new_face_does_not_steal_a_lost_ekf_track():
    cfg = TrackConfig(ekf_max_misses=5)
    base = ScriptedTracker(
        [
            [(10, Box(0, 0, 40, 40))],
            [],
            [(99, Box(300, 300, 340, 340))],
        ]
    )
    tracker = EkfTracker(base, cfg)

    original_id = tracker.update([])[0][0]
    tracker.update([])
    current = tracker.update([])

    assert {track_id for track_id, _ in current} == {original_id, original_id + 1}
