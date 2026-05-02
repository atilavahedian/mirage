from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mirage.agent import DreamerLiteAgent
from mirage.config import MirageConfig
from mirage.envs import ACTION_DIM, OBS_DIM, RocketLanderEnv
from mirage.replay import ReplayBuffer
from mirage.runtime import append_jsonl, ensure_dir, read_json, seed_everything, select_device, timestamp, write_json
from mirage.video import states_to_frames, write_video


def create_run_dir(base_dir: str | Path = "runs", run_name: str = "local") -> Path:
    return ensure_dir(Path(base_dir) / f"{run_name}-{timestamp()}")


def latest_run_dir(base_dir: str | Path = "runs") -> Path | None:
    base = Path(base_dir)
    if not base.exists():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir() and (path / "checkpoints" / "latest.pt").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


class MirageRunner:
    def __init__(self, config: MirageConfig, run_dir: str | Path | None = None, resume: bool = False) -> None:
        seed_everything(config.seed)
        self.config = config
        self.device = select_device(config.device)
        self.run_dir = Path(run_dir) if run_dir else create_run_dir(run_name=config.run_name)
        ensure_dir(self.run_dir)
        ensure_dir(self.run_dir / "checkpoints")
        ensure_dir(self.run_dir / "videos")
        self.env = RocketLanderEnv(
            backend=config.backend,
            seed=config.seed,
            max_episode_steps=config.max_episode_steps,
            render_size=config.render_size,
        )
        self.agent = DreamerLiteAgent(OBS_DIM, ACTION_DIM, config, self.device)
        self.replay = ReplayBuffer(config.replay_capacity, OBS_DIM, ACTION_DIM, seed=config.seed)
        self.obs = self.env.reset(seed=config.seed)
        self.cycle = 0
        self.total_env_steps = 0
        if resume:
            self._resume()
        write_json(self.run_dir / "config.json", config.to_dict())

    def _resume(self) -> None:
        checkpoint = self.run_dir / "checkpoints" / "latest.pt"
        replay_path = self.run_dir / "checkpoints" / "replay.npz"
        state_path = self.run_dir / "runner_state.json"
        if checkpoint.exists():
            extra = self.agent.load(checkpoint)
            self.cycle = int(extra.get("cycle", 0))
            self.total_env_steps = int(extra.get("total_env_steps", 0))
        if replay_path.exists():
            self.replay = ReplayBuffer.load(replay_path, seed=self.config.seed)
        if state_path.exists():
            state = read_json(state_path)
            self.cycle = int(state.get("cycle", self.cycle))
            self.total_env_steps = int(state.get("total_env_steps", self.total_env_steps))

    def run(
        self,
        cycles: int | None = None,
        minutes: float | None = None,
        steps: int | None = None,
    ) -> Path:
        target_cycles = cycles if cycles is not None else 1
        deadline = time.time() + minutes * 60.0 if minutes else None
        target_steps = self.total_env_steps + steps if steps else None
        completed = 0
        while True:
            if target_cycles is not None and completed >= target_cycles:
                break
            if deadline is not None and time.time() >= deadline:
                break
            if target_steps is not None and self.total_env_steps >= target_steps:
                break
            self.run_cycle()
            completed += 1
        return self.run_dir

    def run_cycle(self) -> dict[str, Any]:
        cycle_started = time.time()
        self._write_dashboard_state(phase="acting")
        collect_metrics = self.collect_experience(self.config.steps_per_cycle)
        self._write_dashboard_state(phase="training_world_model")
        world_metrics = self.agent.train_world_model(self.replay, self.config.world_model_steps)
        self._write_dashboard_state(phase="dreaming")
        policy_metrics = self.agent.train_actor_critic(self.replay, self.config.actor_critic_steps)
        self._write_dashboard_state(phase="evaluating")
        eval_metrics, real_frames, dream_frames = self.evaluate(render=True)
        real_video = self.run_dir / "videos" / "real.mp4"
        dream_video = self.run_dir / "videos" / "dream.mp4"
        write_video(real_video, real_frames, fps=self.config.video_fps)
        write_video(dream_video, dream_frames, fps=self.config.video_fps)
        metrics = {
            "cycle": self.cycle,
            "phase": "checkpointed",
            "backend": self.env.backend_name,
            "device": str(self.device),
            "total_env_steps": self.total_env_steps,
            "replay_size": self.replay.size,
            "parameter_count": self.agent.parameter_count(),
            "wall_clock_seconds": time.time() - cycle_started,
            "real_video": str(real_video),
            "dream_video": str(dream_video),
            **collect_metrics,
            **world_metrics,
            **policy_metrics,
            **eval_metrics,
        }
        append_jsonl(self.run_dir / "metrics.jsonl", metrics)
        self.save(metrics)
        self._write_dashboard_state(phase="idle", last_metrics=metrics)
        self.cycle += 1
        return metrics

    def collect_experience(self, steps: int) -> dict[str, Any]:
        episode_return = 0.0
        completed_returns: list[float] = []
        landings = 0
        for _ in range(steps):
            if self.total_env_steps < self.config.warmup_steps:
                action = self.agent.random_action()
            else:
                action = self.agent.act(self.obs)
                action = np.clip(action + np.random.normal(0.0, 0.12, size=action.shape), -1.0, 1.0)
            result = self.env.step(action)
            done = result.terminated or result.truncated
            self.replay.add(self.obs, action, result.reward, done)
            self.obs = result.obs
            episode_return += result.reward
            self.total_env_steps += 1
            if result.info.get("landed"):
                landings += 1
            if done:
                completed_returns.append(episode_return)
                episode_return = 0.0
                self.obs = self.env.reset(seed=self.config.seed + self.total_env_steps)
        return {
            "collect_mean_return": float(np.mean(completed_returns)) if completed_returns else float(episode_return),
            "collect_episodes": len(completed_returns),
            "collect_landings": landings,
        }

    def evaluate(self, render: bool = False) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
        returns = []
        successes = 0
        first_obs = None
        real_frames: list[np.ndarray] = []
        for episode in range(self.config.episodes_per_eval):
            obs = self.env.reset(seed=self.config.seed + 10000 + self.cycle * 31 + episode)
            if first_obs is None:
                first_obs = obs.copy()
            total = 0.0
            for _ in range(self.config.max_episode_steps):
                if render and episode == 0:
                    real_frames.append(self.env.render())
                action = self.agent.act(obs, deterministic=True)
                result = self.env.step(action)
                obs = result.obs
                total += result.reward
                if result.info.get("landed"):
                    successes += 1
                if result.terminated or result.truncated:
                    if render and episode == 0:
                        real_frames.append(self.env.render())
                    break
            returns.append(total)
        dream_states = self.agent.imagine_states(first_obs if first_obs is not None else self.obs, horizon=48)
        dream_frames = states_to_frames(dream_states, size=self.config.render_size)
        dream_error = self._dream_to_real_error(first_obs) if first_obs is not None else 0.0
        return (
            {
                "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
                "eval_return_max": float(np.max(returns)) if returns else 0.0,
                "landing_success_rate": float(successes / max(1, self.config.episodes_per_eval)),
                "dream_to_real_error": dream_error,
            },
            real_frames,
            dream_frames,
        )

    def _dream_to_real_error(self, start_obs: np.ndarray | None) -> float:
        if start_obs is None:
            return 0.0
        probe_env = RocketLanderEnv(
            backend="numpy",
            seed=self.config.seed + self.cycle,
            max_episode_steps=self.config.max_episode_steps,
            render_size=self.config.render_size,
        )
        obs = probe_env.reset(seed=self.config.seed + self.cycle)
        dreamed = self.agent.imagine_states(obs, horizon=12)
        real = []
        for _ in range(len(dreamed)):
            action = self.agent.act(obs, deterministic=True)
            result = probe_env.step(action)
            obs = result.obs
            real.append(obs.copy())
            if result.terminated or result.truncated:
                break
        probe_env.close()
        if not real or len(dreamed) == 0:
            return 0.0
        horizon = min(len(real), len(dreamed))
        return float(np.mean((np.asarray(real[:horizon]) - dreamed[:horizon]) ** 2))

    def save(self, metrics: dict[str, Any] | None = None) -> None:
        extra = {"cycle": self.cycle, "total_env_steps": self.total_env_steps}
        self.agent.save(self.run_dir / "checkpoints" / "latest.pt", extra=extra)
        self.replay.save(self.run_dir / "checkpoints" / "replay.npz")
        write_json(self.run_dir / "runner_state.json", extra)
        if metrics is not None:
            write_json(self.run_dir / "dashboard_state.json", self.dashboard_state("idle", metrics))

    def dashboard_state(self, phase: str, last_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = last_metrics or last_metric_row(self.run_dir / "metrics.jsonl") or {}
        return {
            "name": "Mirage",
            "phase": phase,
            "run_dir": str(self.run_dir),
            "cycle": self.cycle,
            "total_env_steps": self.total_env_steps,
            "replay_size": self.replay.size,
            "device": str(self.device),
            "backend": self.env.backend_name,
            "latest_metrics": metrics,
            "videos": {
                "real": "videos/real.mp4",
                "dream": "videos/dream.mp4",
            },
        }

    def _write_dashboard_state(self, phase: str, last_metrics: dict[str, Any] | None = None) -> None:
        write_json(self.run_dir / "dashboard_state.json", self.dashboard_state(phase, last_metrics))


def last_metric_row(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(rows[-1]) if rows else None
