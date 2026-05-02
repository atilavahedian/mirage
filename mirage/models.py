from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WorldModelOutput:
    obs_pred: torch.Tensor
    reward_pred: torch.Tensor
    continuation_logit: torch.Tensor
    latents: torch.Tensor
    next_latents: torch.Tensor


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dynamics = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continuation_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([latent, action], dim=-1), latent)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> WorldModelOutput:
        batch, steps, _ = obs.shape
        latent = self.encode(obs[:, 0])
        latents = []
        next_latents = []
        obs_preds = []
        reward_preds = []
        continuation_logits = []
        for t in range(steps - 1):
            latents.append(latent)
            latent = self.next_latent(latent, actions[:, t])
            next_latents.append(latent)
            obs_preds.append(self.decoder(latent))
            reward_preds.append(self.reward_head(latent).squeeze(-1))
            continuation_logits.append(self.continuation_head(latent).squeeze(-1))
        return WorldModelOutput(
            obs_pred=torch.stack(obs_preds, dim=1),
            reward_pred=torch.stack(reward_preds, dim=1),
            continuation_logit=torch.stack(continuation_logits, dim=1),
            latents=torch.stack(latents, dim=1),
            next_latents=torch.stack(next_latents, dim=1),
        )

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(obs, actions)
        obs_loss = F.mse_loss(out.obs_pred, obs[:, 1:])
        reward_loss = F.mse_loss(out.reward_pred, rewards[:, :-1])
        continuation_target = 1.0 - dones[:, :-1]
        continuation_loss = F.binary_cross_entropy_with_logits(
            out.continuation_logit, continuation_target
        )
        latent_smooth = 1e-4 * out.next_latents.pow(2).mean()
        total = obs_loss + reward_loss + 0.2 * continuation_loss + latent_smooth
        return {
            "world_model_loss": total,
            "obs_loss": obs_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continuation_loss": continuation_loss.detach(),
        }


class Actor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(latent)
        mean = self.mean(hidden).clamp(-5.0, 5.0)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return mean, log_std

    def sample(self, latent: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(latent)
        if deterministic:
            raw = mean
            log_prob = torch.zeros(mean.shape[:-1], device=mean.device)
        else:
            std = log_std.exp()
            eps = torch.randn_like(mean)
            raw = mean + eps * std
            gaussian_log_prob = -0.5 * (((raw - mean) / std).pow(2) + 2 * log_std + math.log(2 * math.pi))
            log_prob = gaussian_log_prob.sum(dim=-1)
        action = torch.tanh(raw)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
