"""Stable-Baselines3 PPO and A2C training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import A2C, PPO

from training.common import TrainingMetricsCallback, evaluate_policy_on_scenario, make_env, save_json, set_global_seed


def train_ppo(config: dict[str, Any], run_dir: Path, total_timesteps: int, seed: int) -> dict[str, float]:
    set_global_seed(seed)
    env = make_env(seed)
    callback = TrainingMetricsCallback()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        n_epochs=6,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    callback.save(run_dir)
    metrics = evaluate_policy_on_scenario(model, seed=20_000 + seed)
    save_json(run_dir / "result.json", {"algorithm": "ppo", "seed": seed, "config": config, "metrics": metrics})
    env.close()
    return metrics


def train_a2c(config: dict[str, Any], run_dir: Path, total_timesteps: int, seed: int) -> dict[str, float]:
    set_global_seed(seed)
    env = make_env(seed)
    callback = TrainingMetricsCallback()
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    callback.save(run_dir)
    metrics = evaluate_policy_on_scenario(model, seed=20_000 + seed)
    save_json(run_dir / "result.json", {"algorithm": "a2c", "seed": seed, "config": config, "metrics": metrics})
    env.close()
    return metrics

