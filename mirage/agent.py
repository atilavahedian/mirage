from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from mirage.config import MirageConfig
from mirage.models import Actor, Critic, WorldModel, count_parameters
from mirage.replay import ReplayBuffer


@dataclass
class AgentMetrics:
    world_model_loss: float = 0.0
    obs_loss: float = 0.0
    reward_loss: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    imagined_value: float = 0.0


class DreamerLiteAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: MirageConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        self.actor = Actor(config.latent_dim, action_dim, config.hidden_dim).to(device)
        self.critic = Critic(config.latent_dim, config.hidden_dim).to(device)
        self.world_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=config.learning_rate)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.learning_rate)
        self.total_updates = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.world_model.eval()
        self.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent = self.world_model.encode(obs_tensor)
            action, _ = self.actor.sample(latent, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def random_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

    def train_world_model(self, replay: ReplayBuffer, steps: int) -> dict[str, float]:
        if not replay.can_sample(self.config.sequence_length, self.config.batch_size):
            return {"world_model_loss": 0.0, "obs_loss": 0.0, "reward_loss": 0.0}
        totals: dict[str, float] = {}
        self.world_model.train()
        for _ in range(steps):
            batch = replay.sample_sequences(self.config.batch_size, self.config.sequence_length, self.device)
            losses = self.world_model.loss(batch.obs, batch.actions, batch.rewards, batch.dones)
            loss = losses["world_model_loss"]
            self.world_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.world_optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            self.total_updates += 1
        return {key: value / steps for key, value in totals.items()}

    def train_actor_critic(self, replay: ReplayBuffer, steps: int) -> dict[str, float]:
        if replay.size < self.config.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "imagined_value": 0.0}
        totals = {"actor_loss": 0.0, "critic_loss": 0.0, "imagined_value": 0.0}
        for _ in range(steps):
            start_obs = replay.sample_observations(self.config.batch_size, self.device)
            actor_loss, critic_loss, imagined_value = self._imagined_update(start_obs)
            totals["actor_loss"] += actor_loss
            totals["critic_loss"] += critic_loss
            totals["imagined_value"] += imagined_value
        return {key: value / steps for key, value in totals.items()}

    def _imagined_update(self, start_obs: torch.Tensor) -> tuple[float, float, float]:
        for param in self.world_model.parameters():
            param.requires_grad_(False)
        self.actor.train()
        self.critic.train()
        latent = self.world_model.encode(start_obs)
        rewards = []
        critic_latents = []
        log_probs = []
        for _ in range(self.config.imagination_horizon):
            action, log_prob = self.actor.sample(latent)
            latent = self.world_model.next_latent(latent, action)
            reward = self.world_model.reward_head(latent).squeeze(-1)
            rewards.append(reward)
            critic_latents.append(latent.detach())
            log_probs.append(log_prob)
        bootstrap = self.critic(latent).detach()
        returns = []
        running = bootstrap
        for reward in reversed(rewards):
            running = reward + self.config.discount * running
            returns.append(running)
        returns = list(reversed(returns))
        returns_tensor = torch.stack(returns, dim=1)
        log_prob_tensor = torch.stack(log_probs, dim=1)
        actor_objective = returns_tensor.mean(dim=1)
        actor_loss = -actor_objective.mean() - self.config.entropy_scale * (-log_prob_tensor.mean())
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
        values_tensor = torch.stack([self.critic(z) for z in critic_latents], dim=1)
        critic_loss = F.mse_loss(values_tensor, returns_tensor.detach())
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()
        for param in self.world_model.parameters():
            param.requires_grad_(True)
        return (
            float(actor_loss.detach().cpu()),
            float(critic_loss.detach().cpu()),
            float(returns_tensor.mean().detach().cpu()),
        )

    def imagine_states(self, start_obs: np.ndarray, horizon: int) -> np.ndarray:
        self.world_model.eval()
        self.actor.eval()
        states = []
        with torch.no_grad():
            obs_tensor = torch.as_tensor(start_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent = self.world_model.encode(obs_tensor)
            for _ in range(horizon):
                action, _ = self.actor.sample(latent, deterministic=True)
                latent = self.world_model.next_latent(latent, action)
                decoded = self.world_model.decoder(latent).squeeze(0)
                states.append(decoded.cpu().numpy().astype(np.float32))
        return np.stack(states) if states else np.zeros((0, self.obs_dim), dtype=np.float32)

    def parameter_count(self) -> int:
        return (
            count_parameters(self.world_model)
            + count_parameters(self.actor)
            + count_parameters(self.critic)
        )

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "world_optimizer": self.world_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_updates": self.total_updates,
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.world_optimizer.load_state_dict(checkpoint["world_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_updates = int(checkpoint.get("total_updates", 0))
        return dict(checkpoint.get("extra", {}))
