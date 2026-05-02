import numpy as np
import torch

from mirage.agent import DreamerLiteAgent
from mirage.config import MirageConfig


def test_agent_checkpoint_roundtrip(tmp_path) -> None:
    config = MirageConfig(latent_dim=16, hidden_dim=32)
    agent = DreamerLiteAgent(obs_dim=8, action_dim=2, config=config, device=torch.device("cpu"))
    action = agent.act(np.zeros(8, dtype=np.float32), deterministic=True)
    assert action.shape == (2,)
    path = tmp_path / "latest.pt"
    agent.save(path, extra={"cycle": 3})
    restored = DreamerLiteAgent(obs_dim=8, action_dim=2, config=config, device=torch.device("cpu"))
    extra = restored.load(path)
    assert extra["cycle"] == 3
    restored_action = restored.act(np.zeros(8, dtype=np.float32), deterministic=True)
    assert np.allclose(action, restored_action)
