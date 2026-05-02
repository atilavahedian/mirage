from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from mirage.config import MirageConfig, load_config
from mirage.runner import MirageRunner, latest_run_dir, last_metric_row
from mirage.runtime import read_json, write_json


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mirage Lab</title>
  <style>
    :root { color-scheme: light; --bg:#f8f6f1; --ink:#151515; --muted:#706b62; --line:#ded6c9; --accent:#c45f2d; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
    main { width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:34px 0 64px; }
    header { display:flex; justify-content:space-between; gap:20px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:22px; }
    h1 { margin:0; font-family: Georgia, serif; font-size:56px; letter-spacing:-0.04em; }
    .phase { color:var(--accent); font-weight:800; text-transform:uppercase; letter-spacing:0.08em; font-size:12px; }
    .grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:24px; margin-top:24px; }
    section { border-top:1px solid var(--line); padding-top:18px; }
    h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin:0 0 12px; }
    .stats { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; }
    .stat { background:#fffaf3; border:1px solid var(--line); padding:14px; }
    .label { color:var(--muted); font-size:12px; font-weight:700; }
    .value { font-size:24px; font-weight:850; margin-top:6px; }
    .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:18px; }
    .controls label { color:var(--muted); font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; }
    .controls input { width:88px; border:1px solid var(--line); background:#fffaf3; color:var(--ink); padding:11px 12px; font:700 14px/1 Inter, ui-sans-serif, system-ui; }
    button { border:1px solid var(--ink); background:var(--ink); color:#fffaf3; padding:12px 16px; font:800 13px/1 Inter, ui-sans-serif, system-ui; cursor:pointer; }
    button.secondary { background:transparent; color:var(--ink); border-color:var(--line); }
    button:disabled { opacity:0.45; cursor:not-allowed; }
    .control-status { color:var(--muted); font-size:13px; font-weight:700; }
    video { width:100%; background:#ece6dc; border:1px solid var(--line); }
    pre { white-space:pre-wrap; background:#fffaf3; border:1px solid var(--line); padding:16px; color:#302b26; max-height:360px; overflow:auto; }
    @media (max-width: 820px) { .grid, .stats { grid-template-columns:1fr; } h1 { font-size:42px; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class="phase" id="phase">loading</div><h1>Mirage Lab</h1></div>
      <div id="run"></div>
    </header>
    <section>
      <h2>Live State</h2>
      <div class="stats">
        <div class="stat"><div class="label">Cycle</div><div class="value" id="cycle">0</div></div>
        <div class="stat"><div class="label">Steps</div><div class="value" id="steps">0</div></div>
        <div class="stat"><div class="label">Replay</div><div class="value" id="replay">0</div></div>
        <div class="stat"><div class="label">Return</div><div class="value" id="return">0</div></div>
      </div>
      <div class="controls">
        <label for="cycles">Cycles</label>
        <input id="cycles" type="number" min="1" max="1000" value="5" />
        <button id="start-learning">Start learning</button>
        <button class="secondary" id="stop-learning">Stop after current cycle</button>
        <span class="control-status" id="control-status">idle</span>
      </div>
    </section>
    <div class="grid">
      <section><h2>Real Rollout</h2><video id="real-video" controls loop src="/artifacts/videos/real.mp4"></video></section>
      <section><h2>Dream Rollout</h2><video id="dream-video" controls loop src="/artifacts/videos/dream.mp4"></video></section>
    </div>
    <section><h2>Latest Metrics</h2><pre id="metrics">{}</pre></section>
  </main>
  <script>
    let currentVideoVersion = '';

    async function postJSON(url, body = {}) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      return response.json();
    }

    function setVideoSource(id, path, version) {
      const video = document.getElementById(id);
      const src = `/artifacts/${path}?v=${encodeURIComponent(version)}`;
      if (!video.src.endsWith(src)) {
        video.src = src;
      }
    }

    async function refresh() {
      const [state, control] = await Promise.all([
        fetch('/api/state').then(r => r.json()),
        fetch('/api/control').then(r => r.json())
      ]);
      const m = state.latest_metrics || {};
      const running = Boolean(control.running);
      document.getElementById('phase').textContent = state.phase || 'unknown';
      document.getElementById('run').textContent = state.run_dir || '';
      document.getElementById('cycle').textContent = state.cycle ?? 0;
      document.getElementById('steps').textContent = state.total_env_steps ?? 0;
      document.getElementById('replay').textContent = state.replay_size ?? 0;
      document.getElementById('return').textContent = Number(m.eval_return_mean ?? 0).toFixed(1);
      document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
      document.getElementById('start-learning').disabled = running;
      document.getElementById('stop-learning').disabled = !running || Boolean(control.stop_requested);
      document.getElementById('control-status').textContent = running
        ? `learning: ${control.cycles_requested ?? '?'} cycle budget`
        : (control.last_error ? `error: ${control.last_error}` : 'idle');
      const version = `${state.total_env_steps ?? 0}-${m.wall_clock_seconds ?? 0}`;
      if (version !== currentVideoVersion) {
        currentVideoVersion = version;
        setVideoSource('real-video', state.videos?.real || 'videos/real.mp4', version);
        setVideoSource('dream-video', state.videos?.dream || 'videos/dream.mp4', version);
      }
    }

    document.getElementById('start-learning').addEventListener('click', async () => {
      const cycles = Number(document.getElementById('cycles').value || 1);
      document.getElementById('control-status').textContent = 'starting...';
      await postJSON('/api/start', { cycles });
      await refresh();
    });

    document.getElementById('stop-learning').addEventListener('click', async () => {
      await postJSON('/api/stop');
      await refresh();
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


RunnerFactory = Callable[[MirageConfig, str | Path | None, bool], MirageRunner]


class DashboardController:
    def __init__(
        self,
        run_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        runner_cls: RunnerFactory = MirageRunner,
    ) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self.config_path = str(config_path) if config_path else None
        self.runner_cls = runner_cls
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._last_error: str | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._cycles_requested: int | None = None

    def start(self, cycles: int = 5) -> dict[str, Any]:
        cycles = max(1, min(int(cycles), 1000))
        with self._lock:
            if self._is_running_locked():
                status = self._status_locked()
                status["accepted"] = False
                return status
            self._stop_requested.clear()
            self._last_error = None
            self._started_at = time.time()
            self._finished_at = None
            self._cycles_requested = cycles
            self._thread = threading.Thread(target=self._run_learning, args=(cycles,), daemon=True)
            self._thread.start()
            status = self._status_locked()
            status["accepted"] = True
            return status

    def stop(self) -> dict[str, Any]:
        self._stop_requested.set()
        status = self.status()
        status["accepted"] = True
        return status

    def wait(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _is_running_locked(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "name": "Mirage",
            "running": self._is_running_locked(),
            "stop_requested": self._stop_requested.is_set(),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "config_path": self.config_path,
            "cycles_requested": self._cycles_requested,
            "last_error": self._last_error,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
        }

    def _load_runner_config(self) -> MirageConfig:
        if self.config_path:
            return load_config(self.config_path)
        if self.run_dir and (self.run_dir / "config.json").exists():
            return MirageConfig(**read_json(self.run_dir / "config.json"))
        return load_config("configs/smoke.yaml")

    def _run_learning(self, cycles: int) -> None:
        try:
            config = self._load_runner_config()
            resume = self.run_dir is not None and (self.run_dir / "checkpoints" / "latest.pt").exists()
            runner = self.runner_cls(config, self.run_dir, resume)
            self.run_dir = runner.run_dir
            for _ in range(cycles):
                if self._stop_requested.is_set():
                    break
                runner.run(cycles=1)
        except Exception as exc:  # pragma: no cover - surfaced through /api/control for local debugging.
            self._last_error = str(exc)
            if self.run_dir:
                write_json(
                    self.run_dir / "dashboard_state.json",
                    {
                        "name": "Mirage",
                        "phase": "error",
                        "run_dir": str(self.run_dir),
                        "latest_metrics": {"error": self._last_error},
                    },
                )
        finally:
            self._finished_at = time.time()


def load_dashboard_state(run_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(run_dir) if run_dir else latest_run_dir()
    if path is None:
        return {"name": "Mirage", "phase": "no_run", "latest_metrics": {}, "run_dir": None}
    state_path = path / "dashboard_state.json"
    if state_path.exists():
        return read_json(state_path)
    return {
        "name": "Mirage",
        "phase": "unknown",
        "run_dir": str(path),
        "latest_metrics": last_metric_row(path / "metrics.jsonl") or {},
    }


def create_app(run_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Mirage Lab")
    controller = DashboardController(run_dir=run_dir)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(load_dashboard_state(controller.run_dir or run_dir))

    @app.get("/api/control")
    def control() -> JSONResponse:
        return JSONResponse(controller.status())

    @app.post("/api/start")
    def start(payload: dict[str, Any] | None = None) -> JSONResponse:
        payload = payload or {}
        return JSONResponse(controller.start(cycles=int(payload.get("cycles", 5))))

    @app.post("/api/stop")
    def stop() -> JSONResponse:
        return JSONResponse(controller.stop())

    @app.get("/artifacts/{artifact_path:path}", response_model=None)
    def artifacts(artifact_path: str) -> Response:
        state_data = load_dashboard_state(controller.run_dir or run_dir)
        base = state_data.get("run_dir")
        if not base:
            return JSONResponse({"error": "no run available"}, status_code=404)
        base_path = Path(base).resolve()
        path = (base_path / artifact_path).resolve()
        try:
            path.relative_to(base_path)
        except ValueError:
            return JSONResponse({"error": "artifact path escapes run directory"}, status_code=403)
        if not path.exists():
            return JSONResponse({"error": f"missing artifact: {artifact_path}"}, status_code=404)
        return FileResponse(path)

    return app
