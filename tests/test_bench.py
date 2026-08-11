from obscura.bench import coverage, load_annotations
from obscura.geometry import Box


def test_identical_box_is_fully_covered():
    assert coverage(Box(0, 0, 10, 10), [Box(0, 0, 10, 10)]) == 1.0


def test_disjoint_box_covers_nothing():
    assert coverage(Box(0, 0, 10, 10), [Box(50, 50, 60, 60)]) == 0.0


def test_partial_overlap_is_measured():
    assert coverage(Box(0, 0, 10, 10), [Box(0, 0, 5, 10)]) == 0.5


def test_overlapping_boxes_are_not_double_counted():
    # Two boxes each covering the left 60% must not report 120% coverage.
    assert coverage(Box(0, 0, 10, 10), [Box(0, 0, 6, 10), Box(0, 0, 6, 10)]) == 0.6


def test_union_of_partial_boxes_can_reach_full_coverage():
    assert coverage(Box(0, 0, 10, 10), [Box(0, 0, 5, 10), Box(5, 0, 10, 10)]) == 1.0


def test_no_redactions_means_a_full_leak():
    assert coverage(Box(0, 0, 10, 10), []) == 0.0


def test_annotations_load_from_either_shape(tmp_path):
    nested = tmp_path / "nested.json"
    nested.write_text('{"frames": {"0": [[1, 2, 3, 4]]}}', encoding="utf-8")
    flat = tmp_path / "flat.json"
    flat.write_text('{"7": [[1, 2, 3, 4]]}', encoding="utf-8")

    assert load_annotations(nested) == {0: [Box(1, 2, 3, 4)]}
    assert load_annotations(flat) == {7: [Box(1, 2, 3, 4)]}
