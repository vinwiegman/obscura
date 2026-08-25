"""A small FastAPI front end for the scan -> review -> render workflow.

A job pauses at ``review`` with a gallery of the people found in the footage.
The reviewer picks who to protect, and only then is anything rendered.

Jobs live in memory and files in a temp directory, which is the right amount of
machinery for a single-operator tool. Nothing here is safe to expose to a
network you do not control -- see the deployment note in the README.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from ..config import DetectConfig, HealConfig, IdentityConfig, RedactStyle, RunConfig
from ..detect import UnifaceDetector
from ..identity import Person, TrackSampler, tracks_to_redact
from ..pipeline import ScanResult, render_scan, review, scan
from ..track import build as build_tracker
from ..video import VIDEO_SUFFIXES, probe

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GiB
STATIC = Path(__file__).parent / "static"
METHODS = {"blur", "pixelate", "fill"}
SHAPES = {"ellipse", "rect"}
POLICIES = {"except", "only"}

app = FastAPI(title="Obscura", docs_url="/api/docs")

_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # ONNX inference is already thread-hungry
_workspace = Path(tempfile.mkdtemp(prefix="obscura-"))


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"
    """queued | scanning | review | rendering | done | failed"""

    current: int = 0
    total: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    people: list[Person] = field(default_factory=list)
    source: Path | None = None
    output: Path | None = None
    scan_result: ScanResult | None = None
    cfg: RunConfig | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "error": self.error,
            "warnings": self.warnings,
            "summary": self.summary,
            "people": [person.as_dict() for person in self.people],
        }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(file: UploadFile, conf: float = Form(0.5)) -> dict:
    """Upload a video and start the scan. Redaction options come later."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix!r}")

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
    job.cfg = RunConfig(
        detect=DetectConfig(conf=conf),
        heal=HealConfig(),
        identity=IdentityConfig(enabled=True),
    )
    with _lock:
        _jobs[job_id] = job

    _pool.submit(_scan_job, job)
    return job.as_dict()


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict:
    return _get(job_id).as_dict()


@app.get("/api/jobs/{job_id}/people/{person_id}/thumbnail")
def thumbnail(job_id: str, person_id: int) -> Response:
    job = _get(job_id)
    for person in job.people:
        if person.id == person_id and person.thumbnail:
            # Immutable for the life of the job, so let the browser keep it.
            return Response(
                person.thumbnail,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=3600"},
            )
    raise HTTPException(404, "No thumbnail for that person")


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, body: dict = Body(default_factory=dict)) -> dict:
    """Apply a reviewer's selection and render the video."""
    job = _get(job_id)
    if job.status != "review":
        raise HTTPException(409, f"Job is {job.status}, not awaiting review")

    policy = str(body.get("policy", "except"))
    if policy not in POLICIES:
        raise HTTPException(400, f"Unknown policy {policy!r}")
    method = str(body.get("method", "blur"))
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method {method!r}")
    shape = str(body.get("shape", "ellipse"))
    if shape not in SHAPES:
        raise HTTPException(400, f"Unknown shape {shape!r}")

    try:
        selected = {int(value) for value in body.get("selected", [])}
    except (TypeError, ValueError):
        raise HTTPException(400, "selected must be a list of person ids") from None

    known = {person.id for person in job.people}
    if not selected <= known:
        raise HTTPException(400, f"Unknown person ids: {sorted(selected - known)}")

    job.cfg.style = RedactStyle(
        method=method,
        shape=shape,
        strength=float(body.get("strength", 0.35)),
    )
    job.cfg.heal.margin = float(body.get("margin", 0.18))
    job.cfg.keep_audio = bool(body.get("keep_audio", False))

    redact = tracks_to_redact(job.people, selected, policy)
    job.status = "rendering"
    job.current = job.total = 0
    _pool.submit(_render_job, job, redact)
    return job.as_dict()


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


def _progress(job: Job, stage: str) -> callable:
    def report(_stage: str, current: int, total: int) -> None:
        job.status = stage
        job.current, job.total = current, total

    return report


def _scan_job(job: Job) -> None:
    try:
        job.status = "scanning"
        sampler = TrackSampler(job.cfg.identity)
        meta = probe(job.source)
        started = time.perf_counter()
        result = scan(
            job.source,
            UnifaceDetector(job.cfg.detect),
            build_tracker(job.cfg.track),
            meta,
            _progress(job, "scanning"),
            sampler,
        )
        job.scan_result = result
        job.summary = {"scan_seconds": round(time.perf_counter() - started, 2)}
        job.people, job.warnings = review(result, job.cfg, sampler)
        job.status = "review"
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)


def _render_job(job: Job, redact: set[int]) -> None:
    try:
        report = render_scan(
            job.source,
            job.output,
            job.scan_result,
            job.cfg,
            job.summary.get("scan_seconds", 0.0),
            _progress(job, "rendering"),
            redact,
        )
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        return

    job.warnings = job.warnings + report.warnings
    job.summary = {
        "frames": report.meta.n_frames,
        "tracks": report.n_tracks,
        "people": len(job.people),
        "redacted_tracks": len(redact),
        "detections": report.n_detections,
        "redactions": report.n_redactions,
        "healed": report.n_redactions - report.n_detections,
        "fps": round(report.fps, 1),
    }
    job.status = "done"
