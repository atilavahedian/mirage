from __future__ import annotations

import threading
from pathlib import Path

from mirage.config import MirageConfig
from mirage.dashboard import DashboardController, HTML, load_dashboard_state
from mirage.runtime import write_json


def test_dashboard_html_exposes_learning_controls() -> None:
    assert 'id="start-learning"' in HTML
    assert 'id="stop-learning"' in HTML
    assert 'id="control-status"' in HTML


def test_dashboard_controller_starts_learning_run(tmp_path) -> None:
    class FakeRunner:
        def __init__(self, config: MirageConfig, run_dir: str | Path | None = None, resume: bool = False) -> None:
            self.config = config
            self.run_dir = Path(run_dir) if run_dir else tmp_path / "run"
            self.resume = resume
            self.run_dir.mkdir(parents=True, exist_ok=True)

        def run(self, cycles: int | None = None, minutes: float | None = None, steps: int | None = None) -> Path:
            write_json(
                self.run_dir / "dashboard_state.json",
                {
                    "name": "Mirage",
                    "phase": "idle",
                    "run_dir": str(self.run_dir),
                    "cycle": cycles or 0,
                    "total_env_steps": 123,
                    "replay_size": 99,
                    "latest_metrics": {"eval_return_mean": 4.2},
                },
            )
            return self.run_dir

    controller = DashboardController(
        run_dir=tmp_path / "run",
        config_path="configs/smoke.yaml",
        runner_cls=FakeRunner,
    )

    response = controller.start(cycles=2)

    assert response["accepted"] is True
    assert response["running"] is True
    assert controller.wait(timeout=2.0) is True
    status = controller.status()
    assert status["running"] is False
    assert status["run_dir"] == str(tmp_path / "run")
    assert load_dashboard_state(tmp_path / "run")["total_env_steps"] == 123


def test_dashboard_controller_rejects_start_while_running(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class SlowRunner:
        def __init__(self, config: MirageConfig, run_dir: str | Path | None = None, resume: bool = False) -> None:
            self.run_dir = Path(run_dir) if run_dir else tmp_path / "run"
            self.run_dir.mkdir(parents=True, exist_ok=True)

        def run(self, cycles: int | None = None, minutes: float | None = None, steps: int | None = None) -> Path:
            entered.set()
            release.wait(timeout=2.0)
            return self.run_dir

    controller = DashboardController(
        run_dir=tmp_path / "run",
        config_path="configs/smoke.yaml",
        runner_cls=SlowRunner,
    )

    controller.start(cycles=1)
    assert entered.wait(timeout=2.0)
    second = controller.start(cycles=1)
    release.set()
    assert controller.wait(timeout=2.0) is True

    assert second["accepted"] is False
    assert second["running"] is True
