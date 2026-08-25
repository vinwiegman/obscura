"""Shared value types.

``Detection`` carries landmarks alongside the box because face recognition needs
them: an embedding taken from an unaligned crop clusters far worse than one
taken from a crop warped onto canonical eye/nose/mouth positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Box


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float
    landmarks: np.ndarray | None = None
    """5x2 array of (x, y) points in frame coordinates, or None."""

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and self.landmarks.shape == (5, 2)
