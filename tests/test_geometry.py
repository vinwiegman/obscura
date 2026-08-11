from obscura.geometry import Box, iou


def test_expand_scales_with_box_size():
    grown = Box(100, 100, 200, 200).expand(0.1)
    assert (grown.x1, grown.y1, grown.x2, grown.y2) == (90, 90, 210, 210)


def test_top_extra_only_grows_upward():
    grown = Box(0, 100, 100, 200).expand(0.0, top_extra=0.5)
    assert grown.y1 == 50
    assert (grown.x1, grown.x2, grown.y2) == (0, 100, 200)


def test_clip_keeps_box_inside_frame():
    clipped = Box(-20, -20, 50, 50).clip(40, 30)
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0, 0, 40, 30)


def test_clip_fully_outside_yields_empty():
    assert Box(200, 200, 260, 260).clip(100, 100).is_empty


def test_to_int_rounds_outward():
    # Rounding inward would leave a rim of original pixels around the redaction.
    assert Box(10.4, 10.6, 20.1, 20.9).to_int() == (10, 10, 21, 21)


def test_lerp_endpoints_and_midpoint():
    a, b = Box(0, 0, 10, 10), Box(20, 0, 30, 10)
    assert Box.lerp(a, b, 0.0) == a
    assert Box.lerp(a, b, 1.0) == b
    assert Box.lerp(a, b, 0.5) == Box(10, 0, 20, 10)


def test_iou_of_identical_and_disjoint_boxes():
    box = Box(0, 0, 10, 10)
    assert iou(box, box) == 1.0
    assert iou(box, Box(50, 50, 60, 60)) == 0.0
