import numpy as np

from mirage.envs import RocketLanderEnv


def test_numpy_lander_is_deterministic() -> None:
    env_a = RocketLanderEnv(backend="numpy", seed=123, max_episode_steps=10)
    env_b = RocketLanderEnv(backend="numpy", seed=123, max_episode_steps=10)
    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)
    assert np.allclose(obs_a, obs_b)
    action = np.array([0.3, -0.2], dtype=np.float32)
    next_a = env_a.step(action)
    next_b = env_b.step(action)
    assert np.allclose(next_a.obs, next_b.obs)
    assert next_a.reward == next_b.reward


def test_numpy_lander_render_shape() -> None:
    env = RocketLanderEnv(backend="numpy", seed=0, render_size=64)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
