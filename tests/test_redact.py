import cv2
import numpy as np

from obscura import redact
from obscura.config import RedactStyle
from obscura.geometry import Box


def noisy_frame(height: int = 120, width: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_pixels_outside_the_box_are_untouched():
    frame = noisy_frame()
    original = frame.copy()

    redact.apply(frame, [Box(20, 20, 60, 60)], RedactStyle())

    assert np.array_equal(frame[80:, 80:], original[80:, 80:])


def test_fill_replaces_the_region_exactly():
    frame = noisy_frame()
    style = RedactStyle(method="fill", shape="rect", feather=0.0, color=(0, 0, 0))

    redact.apply(frame, [Box(20, 20, 60, 60)], style)

    assert not frame[20:60, 20:60].any()


def test_blur_destroys_local_detail():
    frame = noisy_frame()
    before = float(frame[20:60, 20:60].std())

    redact.apply(frame, [Box(20, 20, 60, 60)], RedactStyle(method="blur", shape="rect"))

    assert float(frame[30:50, 30:50].std()) < before / 3


def test_blur_strength_scales_with_box_size():
    """A kernel that erases a distant face must not leave a close-up one readable."""
    small, large = noisy_frame(), noisy_frame()
    style = RedactStyle(method="blur", shape="rect", feather=0.0)

    redact.apply(small, [Box(10, 10, 30, 30)], style)
    redact.apply(large, [Box(10, 10, 150, 110)], style)

    # Both regions should be flattened to a comparable degree despite the size gap.
    assert float(small[15:25, 15:25].std()) < 25
    assert float(large[50:100, 50:100].std()) < 25


def test_large_faces_never_reach_a_huge_gaussian_kernel(monkeypatch):
    """Guards the fix for a 200x slowdown.

    Sigma must scale with the face, but applying it directly needs a kernel
    hundreds of taps wide: a 700px face cost 4.5 seconds per frame. The blur is
    done on a downscaled copy instead, so the kernel stays small no matter how
    big the face is.
    """
    seen = []
    real = cv2.GaussianBlur

    def spy(src, ksize, sigma, *args, **kwargs):
        seen.append(ksize[0])
        return real(src, ksize, sigma, *args, **kwargs)

    monkeypatch.setattr(redact.cv2, "GaussianBlur", spy)
    frame = noisy_frame(900, 900)

    redact.apply(frame, [Box(0, 0, 800, 800)], RedactStyle(method="blur", shape="rect"))

    assert seen, "no blur was applied"
    assert max(seen) <= 33, f"kernel grew to {max(seen)} taps"


def test_a_huge_face_is_still_thoroughly_destroyed():
    """Speed must not have been bought with a weaker blur."""
    frame = noisy_frame(900, 900)

    redact.apply(frame, [Box(0, 0, 800, 800)], RedactStyle(method="blur", shape="rect"))

    assert float(frame[200:600, 200:600].std()) < 5


def test_pixelate_collapses_the_region_to_few_colours():
    frame = noisy_frame()
    style = RedactStyle(method="pixelate", shape="rect", feather=0.0, strength=0.35)

    redact.apply(frame, [Box(20, 20, 100, 100)], style)

    region = frame[20:100, 20:100].reshape(-1, 3)
    assert len({tuple(pixel) for pixel in region}) < 64


def test_ellipse_covers_the_centre_of_the_box():
    frame = noisy_frame()
    style = RedactStyle(method="fill", shape="ellipse", color=(0, 0, 0))

    redact.apply(frame, [Box(20, 20, 100, 100)], style)

    assert frame[55:65, 55:65].max() == 0


def test_boxes_are_clipped_rather_than_raising():
    frame = noisy_frame()
    redact.apply(frame, [Box(-50, -50, 40, 40), Box(140, 100, 400, 400)], RedactStyle())
    assert frame.shape == (120, 160, 3)


def test_degenerate_boxes_are_skipped():
    frame = noisy_frame()
    original = frame.copy()

    redact.apply(frame, [Box(10, 10, 11, 11), Box(50, 50, 50, 50)], RedactStyle())

    assert np.array_equal(frame, original)


def test_empty_box_list_is_a_no_op():
    frame = noisy_frame()
    original = frame.copy()
    redact.apply(frame, [], RedactStyle())
    assert np.array_equal(frame, original)
