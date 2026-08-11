"""Face detection, adapted from uniface to this tool's box type.

uniface is imported lazily: the geometry, healing and rendering code is useful
and testable without onnxruntime or downloaded weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import DetectConfig
from .geometry import Box

MODELS = {
    "retinaface": ("uniface.detection", "RetinaFace"),
    "scrfd": ("uniface.detection", "SCRFD"),
    "yolov5": ("uniface.detection", "YOLOv5Face"),
    "yolov8": ("uniface.detection", "YOLOv8Face"),
}


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[tuple[Box, float]]: ...


class UnifaceDetector:
    """Wraps a uniface detector, returning ``(box, score)`` pairs."""

    def __init__(self, cfg: DetectConfig) -> None:
        self._cfg = cfg
        self._model = _build(cfg)

    def detect(self, frame: np.ndarray) -> list[tuple[Box, float]]:
        results = []
        for face in self._model.detect(frame):
            # The model already applies the threshold; re-checking costs nothing
            # and keeps the contract true if a detector ignores the kwarg.
            score = float(getattr(face, "confidence", 1.0))
            if score < self._cfg.conf:
                continue
            results.append((_to_box(face), score))
        return results


def _build(cfg: DetectConfig):
    from importlib import import_module

    key = cfg.model.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown detector {cfg.model!r}. Choose from: {', '.join(MODELS)}")

    module_name, class_name = MODELS[key]
    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "uniface is not installed. Install an inference backend:\n"
            "  pip install 'obscura[cpu]'   # CPU / Apple Silicon\n"
            "  pip install 'obscura[gpu]'   # NVIDIA CUDA"
        ) from exc

    try:
        factory = getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - guards a uniface version bump
        raise ImportError(
            f"This build of uniface has no {class_name}. Available: "
            f"{', '.join(sorted(n for n in dir(module) if n[0].isupper()))}"
        ) from exc

    # Threshold inside the model so low-scoring boxes never reach NMS.
    kwargs = {"confidence_threshold": cfg.conf}
    if cfg.providers:
        kwargs["providers"] = cfg.providers
    return factory(**kwargs)


def _to_box(face) -> Box:
    """Read a box off a uniface ``Face``.

    ``bbox_xyxy`` is the documented accessor, but some detectors hand back a
    5-element bbox with the score appended, so the raw attribute is trimmed.
    """
    raw = getattr(face, "bbox_xyxy", None)
    if raw is None:
        raw = face.bbox
    values = np.asarray(raw, dtype=float).ravel()[:4]
    if values.size < 4:
        raise ValueError(f"Detector returned a malformed bbox: {raw!r}")
    x1, y1, x2, y2 = values
    return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
