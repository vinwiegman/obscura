"""Streaming video I/O.

Frames are read and written one at a time. Loading a video into memory is fine
for a 10-second clip and fatal for the hour of 1080p footage this tool is
actually aimed at.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}


class VideoError(RuntimeError):
    pass


@dataclass(slots=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def probe(path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        return VideoMeta(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Container metadata lies often enough to be worth a fallback.
            fps=fps if fps and fps > 0 else 25.0,
            n_frames=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        capture.release()


def frames(path: Path) -> Iterator[np.ndarray]:
    """Yield frames in order."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def count_frames(path: Path) -> int:
    """Exact frame count, by decoding.

    ``CAP_PROP_FRAME_COUNT`` is derived from container metadata and is routinely
    wrong for variable-frame-rate and truncated files. The healing pass indexes
    by frame number, so it needs the real count.
    """
    return sum(1 for _ in frames(path))


@contextmanager
def writer(path: Path, meta: VideoMeta, fourcc: str | None = None):
    """Open a VideoWriter for ``path``, matching the source geometry."""
    code = fourcc or _default_fourcc(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), meta.fps, meta.size)
    if not sink.isOpened():
        raise VideoError(f"Cannot open writer for {path} with fourcc {code!r}")
    try:
        yield sink
    finally:
        sink.release()


def _default_fourcc(path: Path) -> str:
    return "MJPG" if path.suffix.lower() == ".avi" else "mp4v"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_has_audio(path: Path) -> bool:
    """True when the source is *known* to carry audio.

    OpenCV cannot see audio streams at all, so this shells out to ffprobe and
    returns False when ffprobe is missing. False therefore means "no audio, or
    no way to tell" -- it is used to raise a warning, never to skip work.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False
    result = subprocess.run(
        [
            ffprobe,
            "-loglevel",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def remux_audio(source: Path, silent: Path, destination: Path) -> bool:
    """Copy the audio track from ``source`` onto ``silent``.

    OpenCV writes video only. Returns False when ffmpeg is unavailable or the
    source has no audio, leaving the silent file for the caller to keep.
    """
    if not has_ffmpeg():
        return False
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent),
        "-i",
        str(source),
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(destination),
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


def find_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
