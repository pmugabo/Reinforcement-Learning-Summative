from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from environment.custom_env import AquaForecastEnv


def test_environment_is_sb3_compatible() -> None:
    check_env(AquaForecastEnv(), warn=True, skip_render_check=True)


def test_all_actions_produce_valid_observations() -> None:
    for action in range(len(AquaForecastEnv.ACTIONS)):
        env = AquaForecastEnv()
        obs, _ = env.reset(seed=100 + action)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert info["action_id"] == action
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_seeded_resets_are_reproducible() -> None:
    env_a, env_b = AquaForecastEnv(), AquaForecastEnv()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)


def test_episode_has_terminal_or_time_limit() -> None:
    env = AquaForecastEnv(episode_days=12)
    env.reset(seed=7)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps <= 12

