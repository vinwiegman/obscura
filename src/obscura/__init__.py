"""Obscura -- track-aware face anonymization for video."""

from .config import DetectConfig, HealConfig, RedactStyle, RunConfig, TrackConfig
from .geometry import Box
from .pipeline import RunReport, ScanResult, process, scan
from .timeline import TrackTimeline

__version__ = "0.1.0"

__all__ = [
    "Box",
    "DetectConfig",
    "HealConfig",
    "RedactStyle",
    "RunConfig",
    "RunReport",
    "ScanResult",
    "TrackConfig",
    "TrackTimeline",
    "process",
    "scan",
]
