"""Pixel operations that destroy facial detail inside a region."""

from __future__ import annotations

import cv2
import numpy as np

from .config import RedactStyle
from .geometry import Box


def apply(frame: np.ndarray, boxes: list[Box], style: RedactStyle) -> np.ndarray:
    """Redact every box in ``boxes``, in place."""
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.clip(width, height).to_int()
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        obscured = _transform(roi, style)
        frame[y1:y2, x1:x2] = _composite(roi, obscured, style)
    return frame


def _transform(roi: np.ndarray, style: RedactStyle) -> np.ndarray:
    if style.method == "fill":
        return np.full_like(roi, style.color)
    if style.method == "pixelate":
        return _pixelate(roi, style.strength)
    return _blur(roi, style.strength)


MAX_SIGMA = 4.0
"""Largest sigma applied directly; anything heavier is done by downscaling."""


def _blur(roi: np.ndarray, strength: float) -> np.ndarray:
    """Gaussian blur with sigma proportional to region size.

    Sigma has to scale with the face -- a fixed kernel that erases a distant
    face leaves a close-up one readable. But applying that sigma directly needs
    a kernel hundreds of taps wide, and the cost grows with region area times
    kernel width: a 400px face measured at 841ms per frame, which is an hour of
    blurring on a three-minute clip.

    Downscaling first, blurring small, and scaling back is visually equivalent
    for blurs this heavy -- the detail being destroyed is well below the
    sampling grid either way -- and costs a fraction of a millisecond.
    """
    height, width = roi.shape[:2]
    sigma = max(1.0, strength * max(width, height))

    scale = max(1.0, sigma / MAX_SIGMA)
    small = cv2.resize(
        roi,
        (max(1, int(width / scale)), max(1, int(height / scale))),
        interpolation=cv2.INTER_AREA,
    )
    small_sigma = sigma / scale
    ksize = int(small_sigma * 4) | 1
    blurred = cv2.GaussianBlur(small, (ksize, ksize), small_sigma)
    if blurred.shape[:2] == (height, width):
        return blurred
    return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def _pixelate(roi: np.ndarray, strength: float) -> np.ndarray:
    """Downsample to a handful of blocks, then nearest-neighbour back up."""
    height, width = roi.shape[:2]
    blocks = max(1, int(round(1.0 / max(strength, 1e-3) / 8)))
    small = cv2.resize(
        roi,
        (max(1, min(width, blocks)), max(1, min(height, blocks))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _composite(roi: np.ndarray, obscured: np.ndarray, style: RedactStyle) -> np.ndarray:
    """Blend ``obscured`` over ``roi`` using the configured mask shape."""
    if style.shape == "rect" and style.feather <= 0:
        return obscured

    mask = _mask(roi.shape[:2], style)
    if roi.ndim == 3:
        mask = mask[:, :, None]
    return (roi * (1.0 - mask) + obscured * mask).astype(roi.dtype)


MASK_MAX = 160
"""Mask resolution cap. It is a smooth shape, so building it larger buys nothing."""


def _mask(shape: tuple[int, int], style: RedactStyle) -> np.ndarray:
    """Soft-edged coverage mask in [0, 1]."""
    height, width = shape
    # Feathering a full-size mask dominates the cost of the cheap methods; on a
    # 700px face it was most of the 38ms. Build it small and scale it up.
    scale = max(1.0, max(height, width) / MASK_MAX)
    small_h = max(2, int(height / scale))
    small_w = max(2, int(width / scale))
    mask = np.zeros((small_h, small_w), dtype=np.float32)

    if style.shape == "ellipse":
        cv2.ellipse(
            mask,
            center=(small_w // 2, small_h // 2),
            axes=(max(1, small_w // 2), max(1, small_h // 2)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )
    else:
        mask[:, :] = 1.0

    if style.feather > 0:
        ksize = int(style.feather * max(small_w, small_h)) | 1
        if ksize > 1:
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            # Feathering an ellipse pulls its edge inward; push it back out so the
            # fully-opaque core still covers the detected box.
            mask = np.clip(mask * 1.15, 0.0, 1.0)

    if (small_h, small_w) != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask
