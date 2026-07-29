from __future__ import annotations

import json

from environment.custom_env import AquaForecastEnv
from training.configs import CONFIGS


def test_exactly_ten_experiments_per_algorithm() -> None:
    assert set(CONFIGS) == {"dqn", "reinforce", "ppo", "a2c"}
    assert all(len(configurations) == 10 for configurations in CONFIGS.values())
    assert all(len({str(sorted(config.items())) for config in configurations}) == 10 for configurations in CONFIGS.values())


def test_api_state_is_json_serializable() -> None:
    env = AquaForecastEnv()
    env.reset(seed=42)
    env.step(0)
    encoded = json.dumps(env.api_state())
    assert "AquaForecast" in encoded
    assert "Kimironko" in encoded

