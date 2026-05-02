from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from mirage.runtime import ensure_dir, write_json


def load_metrics(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_runs(runs: str | Path = "runs", output_dir: str | Path = "results") -> Path:
    runs_path = Path(runs)
    output = ensure_dir(output_dir)
    rows = []
    for run in sorted(runs_path.glob("*")) if runs_path.exists() else []:
        if run.is_dir():
            for row in load_metrics(run):
                rows.append({"run": run.name, **row})
    write_json(output / "plot_data.json", {"metrics": rows})
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Mirage runs yet", ha="center", va="center")
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        x = list(range(len(rows)))
        axes[0].plot(x, [row.get("eval_return_mean", 0.0) for row in rows], label="eval return")
        axes[0].plot(x, [row.get("collect_mean_return", 0.0) for row in rows], label="collect return")
        axes[0].set_ylabel("return")
        axes[0].legend()
        axes[1].plot(x, [row.get("world_model_loss", 0.0) for row in rows], label="world model")
        axes[1].plot(x, [row.get("dream_to_real_error", 0.0) for row in rows], label="dream error")
        axes[1].set_ylabel("loss / error")
        axes[1].set_xlabel("logged cycle")
        axes[1].legend()
        fig.suptitle("Mirage continual world-model learning")
    fig.tight_layout()
    out = output / "mirage_training.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
