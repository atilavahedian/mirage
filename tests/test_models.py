import torch

from mirage.config import MirageConfig
from mirage.models import Actor, Critic, WorldModel


def test_world_model_output_shapes() -> None:
    model = WorldModel(obs_dim=8, action_dim=2, latent_dim=16, hidden_dim=32)
    obs = torch.randn(4, 6, 8)
    actions = torch.randn(4, 6, 2).clamp(-1, 1)
    out = model(obs, actions)
    assert out.obs_pred.shape == (4, 5, 8)
    assert out.reward_pred.shape == (4, 5)
    assert out.continuation_logit.shape == (4, 5)
    losses = model.loss(obs, actions, torch.randn(4, 6), torch.zeros(4, 6))
    assert losses["world_model_loss"].ndim == 0


def test_actor_and_critic_contracts() -> None:
    actor = Actor(latent_dim=16, action_dim=2, hidden_dim=32)
    critic = Critic(latent_dim=16, hidden_dim=32)
    latent = torch.randn(5, 16)
    action, log_prob = actor.sample(latent)
    value = critic(latent)
    assert action.shape == (5, 2)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert log_prob.shape == (5,)
    assert value.shape == (5,)
