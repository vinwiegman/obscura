"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, video
from .config import DetectConfig, HealConfig, RedactStyle, RunConfig, TrackConfig
from .detect import MODELS
from .pipeline import RunReport, process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obscura",
        description="Track-aware face anonymization for video datasets and CCTV footage.",
    )
    parser.add_argument("--version", action="version", version=f"obscura {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="redact a video file or a directory of them")
    run.add_argument("input", type=Path, help="video file, or directory to walk")
    run.add_argument("-o", "--output", type=Path, help="output file or directory")

    style = run.add_argument_group("redaction style")
    style.add_argument("--method", choices=["blur", "pixelate", "fill"], default="blur")
    style.add_argument("--shape", choices=["ellipse", "rect"], default="ellipse")
    style.add_argument(
        "--strength", type=float, default=0.35, help="blur sigma / block size, relative to box"
    )
    style.add_argument("--feather", type=float, default=0.08, help="mask edge softness")

    coverage = run.add_argument_group("coverage")
    coverage.add_argument("--margin", type=float, default=0.18, help="box dilation on all sides")
    coverage.add_argument("--top-extra", type=float, default=0.25, help="extra headroom for hair")
    coverage.add_argument("--max-gap", type=int, default=45, help="longest detector gap to bridge")
    coverage.add_argument("--lead", type=int, default=8, help="frames covered before first sight")
    coverage.add_argument("--trail", type=int, default=12, help="frames covered after last sight")
    coverage.add_argument(
        "--single-pass",
        action="store_true",
        help="skip healing and redact raw detections (baseline; leaks frames)",
    )

    model = run.add_argument_group("model")
    model.add_argument("--model", choices=sorted(MODELS), default="retinaface")
    model.add_argument("--conf", type=float, default=0.5, help="detection confidence floor")
    model.add_argument("--gpu", action="store_true", help="prefer CUDA execution provider")

    output = run.add_argument_group("output")
    output.add_argument("--fourcc", help="force a codec, e.g. mp4v, avc1, MJPG")
    output.add_argument("--keep-audio", action="store_true", help="remux audio (needs ffmpeg)")
    output.add_argument("-q", "--quiet", action="store_true")

    serve = sub.add_parser("serve", help="run the drag-and-drop web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    bench = sub.add_parser("bench", help="measure leaked frames against ground-truth boxes")
    bench.add_argument("input", type=Path, help="annotated video file")
    bench.add_argument(
        "-a", "--annotations", type=Path, required=True, help="ground-truth JSON (see README)"
    )
    bench.add_argument(
        "--coverage",
        type=float,
        default=0.9,
        help="fraction of a face that must be obscured to count as covered",
    )
    bench.add_argument("--model", choices=sorted(MODELS), default="retinaface")
    bench.add_argument("--conf", type=float, default=0.5)
    bench.add_argument("--gpu", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else None
    return RunConfig(
        detect=DetectConfig(model=args.model, conf=args.conf, providers=providers),
        track=TrackConfig(),
        heal=HealConfig(
            max_gap=args.max_gap,
            lead=args.lead,
            trail=args.trail,
            margin=args.margin,
            top_extra=args.top_extra,
        ),
        style=RedactStyle(
            method=args.method,
            shape=args.shape,
            strength=args.strength,
            feather=args.feather,
        ),
        single_pass=args.single_pass,
        fourcc=args.fourcc,
        keep_audio=args.keep_audio,
    )


class ConsoleProgress:
    """One rewriting line per stage, on stderr so stdout stays pipeable."""

    def __init__(self, label: str, enabled: bool = True) -> None:
        self._label = label
        self._enabled = enabled and sys.stderr.isatty()
        self._stage: str | None = None

    def __call__(self, stage: str, current: int, total: int) -> None:
        if not self._enabled:
            return
        if stage != self._stage:
            self._stage = stage
            print(file=sys.stderr)
        bar = f"{current}/{total}" if total else str(current)
        pct = f" {100 * current // total:>3}%" if total else ""
        print(f"\r  {self._label} [{stage}] {bar}{pct}", end="", file=sys.stderr, flush=True)

    def done(self) -> None:
        if self._enabled:
            print(file=sys.stderr)


def _output_path(source: Path, root: Path, destination: Path | None, batch: bool) -> Path:
    if destination is None:
        return source.with_name(f"{source.stem}.redacted{source.suffix}")
    if batch:
        return destination / source.relative_to(root).with_name(
            f"{source.stem}.redacted{source.suffix}"
        )
    return destination


def _report_line(report: RunReport) -> str:
    added = report.n_redactions - report.n_detections
    return (
        f"{report.output.name}: {report.meta.n_frames} frames, "
        f"{report.n_tracks} tracks, "
        f"{report.n_redactions} redactions (+{added} healed), "
        f"{report.fps:.1f} fps ({report.realtime_factor:.2f}x realtime)"
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    sources = video.find_videos(args.input)
    if not sources:
        print(f"No video files found in {args.input}", file=sys.stderr)
        return 1

    batch = args.input.is_dir()
    if batch and args.output is None:
        print("A directory input needs -o/--output pointing at a directory.", file=sys.stderr)
        return 1

    # Build the detector once; model load and weight download are not cheap.
    from .detect import UnifaceDetector
    from .track import build as build_tracker

    try:
        detector = UnifaceDetector(cfg.detect)
    except (ImportError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    failures = 0
    for position, source in enumerate(sources, start=1):
        destination = _output_path(source, args.input, args.output, batch)
        label = f"{position}/{len(sources)} {source.name}" if batch else source.name
        progress = ConsoleProgress(label, enabled=not args.quiet)
        try:
            # A fresh tracker per video: track ids and Kalman state must not leak
            # across unrelated footage.
            report = process(source, destination, cfg, detector, build_tracker(cfg.track), progress)
        except Exception as exc:  # keep a batch alive when one file is broken
            progress.done()
            print(f"  {source.name}: FAILED -- {exc}", file=sys.stderr)
            failures += 1
            continue
        progress.done()
        if not args.quiet:
            print(_report_line(report))

    return 1 if failures else 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The web UI needs extra packages: pip install 'obscura[web]'", file=sys.stderr)
        return 1

    from .web.app import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    try:
        report = run_benchmark(
            args.input,
            args.annotations,
            threshold=args.coverage,
            model=args.model,
            conf=args.conf,
            gpu=args.gpu,
        )
    except (ImportError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.format())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "bench":
        return cmd_bench(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
