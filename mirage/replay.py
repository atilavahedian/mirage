from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        self.obs[self.index] = np.asarray(obs, dtype=np.float32)
        self.actions[self.index] = np.asarray(action, dtype=np.float32)
        self.rewards[self.index] = float(reward)
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, sequence_length: int, batch_size: int) -> bool:
        return self.size >= max(sequence_length + 1, batch_size)

    def sample_sequences(self, batch_size: int, sequence_length: int, device: torch.device) -> ReplayBatch:
        if not self.can_sample(sequence_length, batch_size):
            raise ValueError("not enough replay data to sample")
        max_start = self.size - sequence_length
        starts = self.rng.integers(0, max_start, size=batch_size)
        obs = np.stack([self.obs[s : s + sequence_length] for s in starts])
        actions = np.stack([self.actions[s : s + sequence_length] for s in starts])
        rewards = np.stack([self.rewards[s : s + sequence_length] for s in starts])
        dones = np.stack([self.dones[s : s + sequence_length] for s in starts])
        return ReplayBatch(
            obs=torch.as_tensor(obs, device=device),
            actions=torch.as_tensor(actions, device=device),
            rewards=torch.as_tensor(rewards, device=device),
            dones=torch.as_tensor(dones, device=device),
        )

    def sample_observations(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.size < batch_size:
            raise ValueError("not enough replay data to sample observations")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return torch.as_tensor(self.obs[idx], device=device)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            capacity=self.capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            index=self.index,
            size=self.size,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
        )

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "ReplayBuffer":
        data = np.load(path)
        buffer = cls(
            capacity=int(data["capacity"]),
            obs_dim=int(data["obs_dim"]),
            action_dim=int(data["action_dim"]),
            seed=seed,
        )
        buffer.index = int(data["index"])
        buffer.size = int(data["size"])
        buffer.obs[:] = data["obs"]
        buffer.actions[:] = data["actions"]
        buffer.rewards[:] = data["rewards"]
        buffer.dones[:] = data["dones"]
        return buffer
