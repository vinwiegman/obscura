"""Healing is the part of this tool that has to be right, so it gets the coverage."""

import pytest

from obscura.config import HealConfig
from obscura.geometry import Box
from obscura.timeline import TrackTimeline, index_from_detections


@pytest.fixture
def bare() -> HealConfig:
    """No dilation and no extension, isolating the interpolation behaviour."""
    return HealConfig(max_gap=45, lead=0, trail=0, margin=0.0, top_extra=0.0)


def timeline_with(observations: dict[int, Box], track_id: int = 1) -> TrackTimeline:
    timeline = TrackTimeline()
    for frame, box in observations.items():
        timeline.add(frame, track_id, box)
    return timeline


def test_gap_is_bridged_by_linear_interpolation(bare):
    timeline = timeline_with({0: Box(0, 0, 10, 10), 4: Box(40, 0, 50, 10)})

    index = timeline.heal(5, (200, 100), bare)

    assert [len(boxes) for boxes in index] == [1, 1, 1, 1, 1]
    assert index[2][0] == Box(20, 0, 30, 10)


def test_gap_longer_than_max_gap_is_left_alone(bare):
    # Past this point the face has probably left frame, and interpolating would
    # smear a redaction across unrelated pixels.
    bare.max_gap = 2
    timeline = timeline_with({0: Box(0, 0, 10, 10), 6: Box(60, 0, 70, 10)})

    index = timeline.heal(7, (200, 100), bare)

    assert [len(boxes) for boxes in index] == [1, 0, 0, 0, 0, 0, 1]


def test_gap_exactly_at_max_gap_is_bridged(bare):
    bare.max_gap = 2
    timeline = timeline_with({0: Box(0, 0, 10, 10), 3: Box(30, 0, 40, 10)})

    index = timeline.heal(4, (200, 100), bare)

    assert all(len(boxes) == 1 for boxes in index)


def test_track_is_extended_before_and_after_its_sightings(bare):
    bare.lead, bare.trail = 3, 2
    timeline = timeline_with({5: Box(0, 0, 10, 10)})

    index = timeline.heal(10, (100, 100), bare)

    covered = [i for i, boxes in enumerate(index) if boxes]
    assert covered == [2, 3, 4, 5, 6, 7]


def test_extension_is_clamped_to_the_video(bare):
    bare.lead, bare.trail = 10, 10
    timeline = timeline_with({0: Box(0, 0, 10, 10), 1: Box(0, 0, 10, 10)})

    index = timeline.heal(3, (100, 100), bare)

    assert len(index) == 3
    assert all(len(boxes) == 1 for boxes in index)


def test_observations_survive_extension_of_a_neighbouring_track(bare):
    bare.lead, bare.trail = 5, 5
    timeline = timeline_with({4: Box(0, 0, 10, 10)}, track_id=1)
    timeline.add(4, 2, Box(50, 50, 60, 60))

    index = timeline.heal(10, (100, 100), bare)

    assert len(index[4]) == 2


def test_margin_and_clipping_are_applied_to_healed_boxes():
    cfg = HealConfig(max_gap=45, lead=0, trail=0, margin=0.5, top_extra=0.0)
    timeline = timeline_with({0: Box(0, 0, 10, 10)})

    box = timeline.heal(1, (100, 100), cfg)[0][0]

    # Grown by 5px a side, then clipped at the frame edge.
    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 15, 15)


def test_empty_timeline_yields_empty_frames(bare):
    assert TrackTimeline().heal(3, (100, 100), bare) == [[], [], []]


def test_baseline_index_does_not_heal():
    cfg = HealConfig(margin=0.0, top_extra=0.0)
    detections = [[Box(0, 0, 10, 10)], [], [Box(0, 0, 10, 10)]]

    index = index_from_detections(detections, (100, 100), cfg)

    assert [len(boxes) for boxes in index] == [1, 0, 1]


def test_baseline_index_drops_boxes_outside_the_frame():
    cfg = HealConfig(margin=0.0, top_extra=0.0)
    index = index_from_detections([[Box(500, 500, 510, 510)]], (100, 100), cfg)
    assert index == [[]]


def test_ekf_prediction_extends_coverage_without_a_future_detection(bare):
    timeline = timeline_with({0: Box(0, 0, 10, 10)})
    timeline.add(1, 1, Box(5, 0, 15, 10), predicted=True)
    timeline.add(2, 1, Box(10, 0, 20, 10), predicted=True)

    index = timeline.heal(3, (100, 100), bare)

    assert [boxes[0] for boxes in index] == [
        Box(0, 0, 10, 10),
        Box(5, 0, 15, 10),
        Box(10, 0, 20, 10),
    ]


def test_hindsight_interpolation_wins_over_ekf_prediction(bare):
    timeline = timeline_with({0: Box(0, 0, 10, 10), 2: Box(20, 0, 30, 10)})
    timeline.add(1, 1, Box(100, 0, 110, 10), predicted=True)

    index = timeline.heal(3, (200, 100), bare)

    assert index[1][0] == Box(10, 0, 20, 10)


def test_plausible_track_fragments_are_stitched_before_interpolation(bare):
    timeline = timeline_with({0: Box(0, 0, 20, 20), 1: Box(2, 0, 22, 20)}, track_id=1)
    timeline.add(4, 99, Box(8, 0, 28, 20))
    timeline.add(5, 99, Box(10, 0, 30, 20))
    # This deliberately bad forward prediction must lose to observations on
    # both sides once the fragments are stitched.
    timeline.add(2, 1, Box(100, 0, 120, 20), predicted=True)

    index = timeline.heal(6, (200, 100), bare)

    assert [len(boxes) for boxes in index] == [1, 1, 1, 1, 1, 1]
    assert index[2][0] == Box(4, 0, 24, 20)
    assert index[3][0] == Box(6, 0, 26, 20)


def test_selection_covers_every_track_a_stitched_group_absorbed(bare):
    """Stitching runs before the filter, so a group spans several track ids.

    Selecting any one of them must redact the whole group: if motion stitching
    wrongly joined two people, over-blurring is survivable and leaving one
    visible is not.
    """
    timeline = timeline_with({0: Box(0, 0, 20, 20), 1: Box(2, 0, 22, 20)}, track_id=1)
    timeline.add(4, 99, Box(8, 0, 28, 20))
    timeline.add(5, 99, Box(10, 0, 30, 20))

    from_first = timeline.heal(6, (200, 100), bare, only={1})
    from_second = timeline.heal(6, (200, 100), bare, only={99})

    assert [len(boxes) for boxes in from_first] == [1, 1, 1, 1, 1, 1]
    assert [len(boxes) for boxes in from_second] == [1, 1, 1, 1, 1, 1]


def test_selecting_an_unrelated_track_redacts_nothing(bare):
    timeline = timeline_with({0: Box(0, 0, 20, 20), 1: Box(2, 0, 22, 20)}, track_id=1)

    index = timeline.heal(3, (200, 100), bare, only={4242})

    assert all(not boxes for boxes in index)


def test_unstitched_tracks_are_selected_independently(bare):
    """Two people far apart stay separate, so one can be kept visible."""
    timeline = timeline_with({0: Box(0, 0, 20, 20), 1: Box(1, 0, 21, 20)}, track_id=1)
    timeline.add(0, 2, Box(300, 300, 320, 320))
    timeline.add(1, 2, Box(301, 300, 321, 320))

    index = timeline.heal(2, (400, 400), bare, only={2})

    assert [len(boxes) for boxes in index] == [1, 1]
    assert index[0][0].x1 >= 300


def test_spans_include_prediction_only_tracks():
    """A track missing from spans() would belong to no person and never be
    selectable -- a silent way to leak a face."""
    timeline = TrackTimeline()
    timeline.add(3, 7, Box(0, 0, 10, 10), predicted=True)
    timeline.add(4, 7, Box(1, 0, 11, 10), predicted=True)

    spans = timeline.spans()

    assert spans[7] == (2, 3, 4)


def test_spans_count_observed_and_predicted_frames_together():
    timeline = TrackTimeline()
    timeline.add(0, 1, Box(0, 0, 10, 10))
    timeline.add(1, 1, Box(1, 0, 11, 10), predicted=True)

    assert timeline.spans()[1] == (2, 0, 1)


def test_implausibly_distant_later_track_is_not_stitched(bare):
    timeline = timeline_with({0: Box(0, 0, 20, 20)}, track_id=1)
    timeline.add(3, 2, Box(300, 300, 320, 320))

    index = timeline.heal(4, (400, 400), bare)

    assert [len(boxes) for boxes in index] == [1, 0, 0, 1]
