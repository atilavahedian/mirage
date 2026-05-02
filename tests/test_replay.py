import numpy as np
import torch

from mirage.replay import ReplayBuffer


def test_replay_persists_and_samples(tmp_path) -> None:
    replay = ReplayBuffer(capacity=20, obs_dim=8, action_dim=2, seed=5)
    for i in range(12):
        replay.add(np.ones(8) * i, np.ones(2) * 0.1, float(i), done=False)
    path = tmp_path / "replay.npz"
    replay.save(path)
    loaded = ReplayBuffer.load(path, seed=5)
    assert loaded.size == 12
    batch = loaded.sample_sequences(batch_size=3, sequence_length=4, device=torch.device("cpu"))
    assert batch.obs.shape == (3, 4, 8)
    assert batch.actions.shape == (3, 4, 2)
