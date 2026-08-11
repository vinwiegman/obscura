"""A small FastAPI front end: drop a video in, poll progress, download the result.

Jobs live in memory and files in a temp directory, which is the right amount of
machinery for a single-operator tool. Nothing here is safe to expose to a
network you do not control -- see the deployment note in the README.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from ..config import DetectConfig, HealConfig, RedactStyle, RunConfig
from ..pipeline import process
from ..video import VIDEO_SUFFIXES

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GiB
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Obscura", docs_url="/api/docs")

_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # ONNX inference is already thread-hungry
_workspace = Path(tempfile.mkdtemp(prefix="obscura-"))


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"  # queued | scanning | rendering | done | failed
    current: int = 0
    total: int = 0
    error: str | None = None
    summary: dict = field(default_factory=dict)
    source: Path | None = None
    output: Path | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "error": self.error,
            "summary": self.summary,
        }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
    method: str = Form("blur"),
    shape: str = Form("ellipse"),
    margin: float = Form(0.18),
    strength: float = Form(0.35),
    keep_audio: bool = Form(False),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix!r}")
    if method not in {"blur", "pixelate", "fill"}:
        raise HTTPException(400, f"Unknown method {method!r}")
    if shape not in {"ellipse", "rect"}:
        raise HTTPException(400, f"Unknown shape {shape!r}")

    job_id = uuid.uuid4().hex
    directory = _workspace / job_id
    directory.mkdir(parents=True)
    source = directory / f"input{suffix}"

    written = 0
    with source.open("wb") as sink:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                shutil.rmtree(directory, ignore_errors=True)
                raise HTTPException(413, "Upload exceeds the 2 GiB limit")
            sink.write(chunk)

    job = Job(id=job_id, name=file.filename or source.name, source=source)
    job.output = directory / f"redacted{suffix}"
    with _lock:
        _jobs[job_id] = job

    cfg = RunConfig(
        detect=DetectConfig(),
        heal=HealConfig(margin=margin),
        style=RedactStyle(method=method, shape=shape, strength=strength),
        keep_audio=keep_audio,
    )
    _pool.submit(_run, job, cfg)
    return job.as_dict()


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict:
    return _get(job_id).as_dict()


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = _get(job_id)
    if job.status != "done" or job.output is None or not job.output.exists():
        raise HTTPException(409, "Job is not finished")
    stem = Path(job.name).stem
    return FileResponse(
        job.output,
        media_type="application/octet-stream",
        filename=f"{stem}.redacted{job.output.suffix}",
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _get(job_id)
    with _lock:
        _jobs.pop(job_id, None)
    if job.source is not None:
        shutil.rmtree(job.source.parent, ignore_errors=True)
    return {"deleted": job_id}


def _get(job_id: str) -> Job:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job


def _run(job: Job, cfg: RunConfig) -> None:
    def progress(stage: str, current: int, total: int) -> None:
        job.status = "scanning" if stage == "scan" else "rendering"
        job.current, job.total = current, total

    try:
        report = process(job.source, job.output, cfg, progress=progress)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        return

    job.summary = {
        "frames": report.meta.n_frames,
        "tracks": report.n_tracks,
        "detections": report.n_detections,
        "redactions": report.n_redactions,
        "healed": report.n_redactions - report.n_detections,
        "fps": round(report.fps, 1),
        "warnings": report.warnings,
    }
    job.status = "done"
