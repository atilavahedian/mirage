from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from mirage.runner import latest_run_dir, last_metric_row
from mirage.runtime import read_json


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
    </section>
    <div class="grid">
      <section><h2>Real Rollout</h2><video controls loop src="/artifacts/videos/real.mp4"></video></section>
      <section><h2>Dream Rollout</h2><video controls loop src="/artifacts/videos/dream.mp4"></video></section>
    </div>
    <section><h2>Latest Metrics</h2><pre id="metrics">{}</pre></section>
  </main>
  <script>
    async function refresh() {
      const state = await fetch('/api/state').then(r => r.json());
      const m = state.latest_metrics || {};
      document.getElementById('phase').textContent = state.phase || 'unknown';
      document.getElementById('run').textContent = state.run_dir || '';
      document.getElementById('cycle').textContent = state.cycle ?? 0;
      document.getElementById('steps').textContent = state.total_env_steps ?? 0;
      document.getElementById('replay').textContent = state.replay_size ?? 0;
      document.getElementById('return').textContent = Number(m.eval_return_mean ?? 0).toFixed(1);
      document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""


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

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(load_dashboard_state(run_dir))

    @app.get("/artifacts/{artifact_path:path}", response_model=None)
    def artifacts(artifact_path: str) -> Response:
        state_data = load_dashboard_state(run_dir)
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
