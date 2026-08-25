"""Configuration objects for a redaction run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Method = Literal["blur", "pixelate", "fill"]
Shape = Literal["rect", "ellipse"]


@dataclass(slots=True)
class DetectConfig:
    model: str = "retinaface"
    conf: float = 0.5
    """Detection confidence floor.

    Lower than you would pick for recognition. A false positive costs a blurred
    patch of wall; a false negative costs an identifiable face, so the asymmetry
    runs the other way here.
    """
    providers: list[str] | None = None


@dataclass(slots=True)
class TrackConfig:
    track_thresh: float = 0.4
    track_buffer: int = 60
    match_thresh: float = 0.8
    ekf_max_misses: int = 30
    """Frames for which the box EKF may emit a prediction without a detection.

    This is deliberately shorter than the identity track buffer by default.
    Continuing forever would eventually redact unrelated parts of the scene
    after a person has actually left the frame.
    """


@dataclass(slots=True)
class HealConfig:
    """Turns sparse per-frame detections into continuous coverage."""

    max_gap: int = 45
    """Longest run of detector misses to bridge by interpolation.

    Beyond this the face has probably left the scene, and interpolating would
    smear a redaction across unrelated pixels.
    """

    lead: int = 8
    """Frames to cover *before* a track's first detection."""

    trail: int = 12
    """Frames to cover *after* a track's last detection."""

    margin: float = 0.18
    """Box dilation on every side, as a fraction of box size."""

    top_extra: float = 0.25
    """Extra upward dilation, for hair and forehead."""


@dataclass(slots=True)
class RedactStyle:
    method: Method = "blur"
    shape: Shape = "ellipse"
    strength: float = 0.35
    """Blur sigma as a fraction of box width, or inverse block count for pixelate.

    Scaling with box size is the point: a fixed 15px kernel that erases a distant
    face leaves a close-up one perfectly readable.
    """

    feather: float = 0.08
    """Mask edge softness, as a fraction of box size."""

    color: tuple[int, int, int] = (0, 0, 0)
    """BGR fill colour, used by ``method="fill"``."""


@dataclass(slots=True)
class RunConfig:
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    heal: HealConfig = field(default_factory=HealConfig)
    style: RedactStyle = field(default_factory=RedactStyle)
    single_pass: bool = False
    """Skip healing and redact raw per-frame detections.

    Only useful as the baseline in benchmarks; it is the behaviour this tool
    exists to improve on.
    """

    fourcc: str | None = None
    keep_audio: bool = False
