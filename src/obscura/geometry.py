"""Axis-aligned box maths.

Deliberately free of OpenCV and uniface imports so the geometry that matters for
correctness can be unit-tested without models or video files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned box in pixel coordinates, ``x2``/``y2`` exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def expand(self, ratio: float, top_extra: float = 0.0) -> Box:
        """Grow the box by ``ratio`` of its own size on every side.

        ``top_extra`` adds further headroom upward only. Detectors box the face,
        not the head; hair and forehead are identifying and routinely fall
        outside a tight bbox.
        """
        dx = self.width * ratio
        dy = self.height * ratio
        return Box(
            self.x1 - dx,
            self.y1 - dy - self.height * top_extra,
            self.x2 + dx,
            self.y2 + dy,
        )

    def clip(self, width: int, height: int) -> Box:
        """Clamp to a frame of ``width`` x ``height``."""
        return Box(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def to_int(self) -> tuple[int, int, int, int]:
        """Round outward, so the redacted region never falls short of the box."""
        import math

        return (
            int(math.floor(self.x1)),
            int(math.floor(self.y1)),
            int(math.ceil(self.x2)),
            int(math.ceil(self.y2)),
        )

    @staticmethod
    def lerp(a: Box, b: Box, t: float) -> Box:
        """Linear blend between two boxes; ``t=0`` gives ``a``, ``t=1`` gives ``b``."""
        return Box(
            a.x1 + (b.x1 - a.x1) * t,
            a.y1 + (b.y1 - a.y1) * t,
            a.x2 + (b.x2 - a.x2) * t,
            a.y2 + (b.y2 - a.y2) * t,
        )


def intersection_area(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
