from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mirage.envs import render_state


def write_video(path: str | Path, frames: list[np.ndarray], fps: int = 20) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        frames = [np.full((96, 96, 3), 247, dtype=np.uint8)]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def states_to_frames(states: np.ndarray, size: int = 128) -> list[np.ndarray]:
    return [render_state(state, size=size) for state in states]
