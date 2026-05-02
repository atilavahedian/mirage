from pathlib import Path

from mirage.cli import main


def test_smoke_daemon_cycle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    repo_config = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    code = main(["daemon", "--cycles", "1", "--config", str(repo_config), "--run-dir", str(tmp_path / "run")])
    assert code == 0
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    assert (tmp_path / "run" / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "run" / "videos" / "real.mp4").exists()
    assert (tmp_path / "run" / "videos" / "dream.mp4").exists()
    assert (tmp_path / "run" / "dashboard_state.json").exists()
