from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from mirage.config import load_config
from mirage.dashboard import create_app, load_dashboard_state
from mirage.plotting import plot_runs
from mirage.runner import MirageRunner, latest_run_dir
from mirage.runtime import timestamp, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirage", description="Continual world-model rocket lander lab")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="configs/local_mps.yaml")
        p.add_argument("--run-dir")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--cycles", type=int)
        p.add_argument("--minutes", type=float)
        p.add_argument("--steps", type=int)
        p.add_argument("--backend")
        p.add_argument("--device")
        p.add_argument("--seed", type=int)

    add_run_args(sub.add_parser("daemon", help="run persistent act/dream/train cycles"))
    add_run_args(sub.add_parser("train", help="alias for daemon"))

    eval_parser = sub.add_parser("eval", help="evaluate a checkpoint")
    eval_parser.add_argument("--config", default="configs/local_mps.yaml")
    eval_parser.add_argument("--checkpoint")
    eval_parser.add_argument("--run-dir")
    eval_parser.add_argument("--episodes", type=int)

    dash = sub.add_parser("dashboard", help="serve the local research dashboard")
    dash.add_argument("--run-dir")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--once", action="store_true")

    demo = sub.add_parser("demo", help="run a tiny demo cycle")
    demo.add_argument("--config", default="configs/smoke.yaml")

    plot = sub.add_parser("plot", help="plot run metrics")
    plot.add_argument("--runs", default="runs")
    plot.add_argument("--output-dir", default="results")
    return parser


def run_command(args: argparse.Namespace) -> int:
    config = load_config(
        args.config,
        backend=getattr(args, "backend", None),
        device=getattr(args, "device", None),
        seed=getattr(args, "seed", None),
    )
    runner = MirageRunner(config, run_dir=args.run_dir, resume=args.resume)
    run_dir = runner.run(cycles=args.cycles, minutes=args.minutes, steps=args.steps)
    print(f"Mirage run written to {run_dir}")
    return 0


def eval_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.episodes is not None:
        config.episodes_per_eval = args.episodes
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if run_dir is None and args.checkpoint is None:
        raise SystemExit("no run found; pass --checkpoint or --run-dir")
    runner = MirageRunner(config, run_dir=run_dir, resume=bool(run_dir))
    if args.checkpoint:
        runner.agent.load(args.checkpoint)
    metrics, real_frames, dream_frames = runner.evaluate(render=True)
    out_dir = Path("runs") / f"eval-{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    print(f"Mirage eval metrics written to {out_dir / 'metrics.json'}")
    return 0


def dashboard_command(args: argparse.Namespace) -> int:
    if args.once:
        print(load_dashboard_state(args.run_dir))
        return 0
    uvicorn.run(create_app(args.run_dir), host=args.host, port=args.port, log_level="info")
    return 0


def demo_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runner = MirageRunner(config)
    run_dir = runner.run(cycles=1)
    print(f"Demo complete: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"daemon", "train"}:
        return run_command(args)
    if args.command == "eval":
        return eval_command(args)
    if args.command == "dashboard":
        return dashboard_command(args)
    if args.command == "demo":
        return demo_command(args)
    if args.command == "plot":
        out = plot_runs(args.runs, args.output_dir)
        print(f"Plot written to {out}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
