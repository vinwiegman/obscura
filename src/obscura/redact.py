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


def _blur(roi: np.ndarray, strength: float) -> np.ndarray:
    """Gaussian blur with sigma proportional to region size."""
    height, width = roi.shape[:2]
    sigma = max(1.0, strength * max(width, height))
    # Kernel wide enough that the Gaussian is not truncated, and always odd.
    ksize = int(sigma * 4) | 1
    return cv2.GaussianBlur(roi, (ksize, ksize), sigma)


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


def _mask(shape: tuple[int, int], style: RedactStyle) -> np.ndarray:
    """Soft-edged coverage mask in [0, 1]."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)

    if style.shape == "ellipse":
        cv2.ellipse(
            mask,
            center=(width // 2, height // 2),
            axes=(max(1, width // 2), max(1, height // 2)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )
    else:
        mask[:, :] = 1.0

    if style.feather > 0:
        ksize = int(style.feather * max(width, height)) | 1
        if ksize > 1:
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            # Feathering an ellipse pulls its edge inward; push it back out so the
            # fully-opaque core still covers the detected box.
            mask = np.clip(mask * 1.15, 0.0, 1.0)

    return mask
