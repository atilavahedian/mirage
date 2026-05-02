from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


OBS_DIM = 8
ACTION_DIM = 2


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RocketLanderEnv:
    """Continuous rocket lander wrapper with a Gymnasium backend and a NumPy fallback."""

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    def __init__(
        self,
        backend: str = "auto",
        seed: int = 0,
        max_episode_steps: int = 600,
        render_size: int = 128,
    ) -> None:
        self.backend_name = backend
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.render_size = render_size
        self._backend = self._make_backend(backend, seed, max_episode_steps, render_size)

    def _make_backend(
        self, backend: str, seed: int, max_episode_steps: int, render_size: int
    ) -> "GymRocketBackend | NumpyRocketBackend":
        if backend not in {"auto", "gymnasium", "numpy"}:
            raise ValueError(f"unknown backend: {backend}")
        if backend in {"auto", "gymnasium"}:
            try:
                return GymRocketBackend(seed=seed, max_episode_steps=max_episode_steps)
            except Exception as exc:
                if backend == "gymnasium":
                    raise
                self.backend_name = "numpy"
                return NumpyRocketBackend(
                    seed=seed,
                    max_episode_steps=max_episode_steps,
                    render_size=render_size,
                    fallback_reason=str(exc),
                )
        self.backend_name = "numpy"
        return NumpyRocketBackend(seed=seed, max_episode_steps=max_episode_steps, render_size=render_size)

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._backend.reset(seed=seed)

    def step(self, action: np.ndarray) -> StepResult:
        return self._backend.step(action)

    def render(self) -> np.ndarray:
        return self._backend.render()

    def close(self) -> None:
        self._backend.close()


class GymRocketBackend:
    def __init__(self, seed: int, max_episode_steps: int) -> None:
        import gymnasium as gym

        try:
            self.env = gym.make(
                "LunarLander-v3",
                continuous=True,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
            )
        except TypeError:
            self.env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
        self.seed = seed
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.seed if seed is None else seed)
        self.steps = 0
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        )
        self.steps += 1
        return StepResult(
            obs=np.asarray(obs, dtype=np.float32),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def render(self) -> np.ndarray:
        frame = self.env.render()
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self.env.close()


class NumpyRocketBackend:
    """Small deterministic continuous lander used when Box2D is unavailable."""

    def __init__(
        self,
        seed: int,
        max_episode_steps: int,
        render_size: int,
        fallback_reason: str | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.render_size = render_size
        self.fallback_reason = fallback_reason
        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x = self.rng.uniform(-0.65, 0.65)
        y = self.rng.uniform(0.75, 1.15)
        vx = self.rng.uniform(-0.04, 0.04)
        vy = self.rng.uniform(-0.02, 0.02)
        angle = self.rng.uniform(-0.25, 0.25)
        angular_velocity = self.rng.uniform(-0.04, 0.04)
        self.state = np.array([x, y, vx, vy, angle, angular_velocity, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        main = max(float(action[0]), 0.0)
        side = float(action[1])
        x, y, vx, vy, angle, av, _, _ = [float(v) for v in self.state]
        dt = 0.08
        gravity = -0.045
        thrust = 0.105 * main
        lateral = 0.035 * side
        ax = lateral - 0.012 * vx - 0.018 * math.sin(angle) * main
        ay = gravity + thrust * math.cos(angle) - 0.01 * vy
        av += (-0.06 * side - 0.04 * angle) * dt
        angle += av
        vx += ax
        vy += ay
        x += vx * dt * 4.0
        y += vy * dt * 4.0
        leg_contact = y <= 0.03 and abs(angle) < 0.35
        terminated = False
        landed = False
        crashed = False
        if y <= 0.0:
            y = 0.0
            speed = math.sqrt(vx * vx + vy * vy)
            landed = abs(x) < 0.18 and speed < 0.12 and abs(angle) < 0.25
            crashed = not landed
            terminated = True
            vx *= 0.2
            vy = 0.0
            av *= 0.2
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        left_leg = 1.0 if leg_contact else 0.0
        right_leg = 1.0 if leg_contact else 0.0
        self.state = np.array([x, y, vx, vy, angle, av, left_leg, right_leg], dtype=np.float32)
        distance_penalty = 2.0 * abs(x) + 1.5 * abs(y)
        velocity_penalty = 0.7 * math.sqrt(vx * vx + vy * vy)
        angle_penalty = 0.35 * abs(angle)
        fuel_penalty = 0.025 * (main + abs(side))
        reward = -(distance_penalty + velocity_penalty + angle_penalty + fuel_penalty)
        if landed:
            reward += 100.0
        if crashed:
            reward -= 100.0
        return StepResult(
            obs=self.state.copy(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "backend": "numpy",
                "landed": landed,
                "crashed": crashed,
                "fallback_reason": self.fallback_reason,
            },
        )

    def render(self) -> np.ndarray:
        return render_state(self.state, size=self.render_size)

    def close(self) -> None:
        return None


def render_state(state: np.ndarray, size: int = 128) -> np.ndarray:
    canvas = np.full((size, size, 3), 247, dtype=np.uint8)
    ground_y = int(size * 0.82)
    canvas[ground_y:, :, :] = np.array([235, 230, 221], dtype=np.uint8)
    pad_half = int(size * 0.09)
    center_x = size // 2
    canvas[ground_y - 2 : ground_y + 2, center_x - pad_half : center_x + pad_half] = np.array(
        [35, 35, 35], dtype=np.uint8
    )
    x, y, vx, vy, angle, _, _, _ = [float(v) for v in state]
    px = int(np.clip(center_x + x * size * 0.35, 8, size - 9))
    py = int(np.clip(ground_y - y * size * 0.55, 8, ground_y - 5))
    body = np.array(
        [
            [-7, -8],
            [7, -8],
            [9, 8],
            [-9, 8],
        ],
        dtype=np.float32,
    )
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    pts = body @ rot.T + np.array([px, py], dtype=np.float32)
    for i in range(len(pts)):
        draw_line(canvas, pts[i], pts[(i + 1) % len(pts)], color=(20, 20, 20))
    leg_left = np.array([[-6, 8], [-13, 16]], dtype=np.float32) @ rot.T + np.array([px, py])
    leg_right = np.array([[6, 8], [13, 16]], dtype=np.float32) @ rot.T + np.array([px, py])
    draw_line(canvas, leg_left[0], leg_left[1], color=(20, 20, 20))
    draw_line(canvas, leg_right[0], leg_right[1], color=(20, 20, 20))
    if vy > 0.02:
        flame = np.array([[0, 10], [-4, 20], [4, 20]], dtype=np.float32) @ rot.T + np.array([px, py])
        draw_line(canvas, flame[0], flame[1], color=(196, 95, 45))
        draw_line(canvas, flame[0], flame[2], color=(196, 95, 45))
    end_v = np.array([px + vx * 130.0, py - vy * 130.0])
    draw_line(canvas, np.array([px, py]), end_v, color=(75, 105, 155))
    return canvas


def draw_line(canvas: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w, _ = canvas.shape
    x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
    x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            canvas[max(0, y0 - 1) : min(h, y0 + 2), max(0, x0 - 1) : min(w, x0 + 2)] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
