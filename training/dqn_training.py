"""Stable-Baselines3 DQN training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

from training.common import TrainingMetricsCallback, evaluate_policy_on_scenario, make_env, save_json, set_global_seed


def train_dqn(config: dict[str, Any], run_dir: Path, total_timesteps: int, seed: int) -> dict[str, float]:
    set_global_seed(seed)
    env = make_env(seed)
    callback = TrainingMetricsCallback()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        learning_starts=min(500, total_timesteps // 10),
        exploration_initial_eps=1.0,
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        train_freq=4,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    callback.save(run_dir)
    metrics = evaluate_policy_on_scenario(model, seed=20_000 + seed)
    save_json(run_dir / "result.json", {"algorithm": "dqn", "seed": seed, "config": config, "metrics": metrics})
    env.close()
    return metrics

