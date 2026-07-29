"""Shared callbacks, evaluation, persistence, and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import AquaForecastEnv


ROOT = Path(__file__).resolve().parents[1]


class Predictor(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]: ...


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_env(seed: int | None = None, scenario: str = "normal") -> Monitor:
    env = AquaForecastEnv()
    env.reset(seed=seed, options={"scenario": scenario})
    return Monitor(env)


class TrainingMetricsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episodes: list[dict[str, float]] = []
        self.training: list[dict[str, float]] = []
        self._episode_number = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_number += 1
                ep = info["episode"]
                self.episodes.append(
                    {
                        "episode": self._episode_number,
                        "timesteps": float(self.num_timesteps),
                        "reward": float(ep["r"]),
                        "length": float(ep["l"]),
                    }
                )
        if self.num_timesteps % 100 == 0:
            values = self.logger.name_to_value
            row = {"timesteps": float(self.num_timesteps)}
            if "train/loss" in values:
                row["loss"] = float(values["train/loss"])
            if "train/entropy_loss" in values:
                row["entropy"] = -float(values["train/entropy_loss"])
            if "train/exploration_rate" in values:
                row["exploration_rate"] = float(values["train/exploration_rate"])
            if len(row) > 1:
                self.training.append(row)
        return True

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.episodes).to_csv(directory / "episodes.csv", index=False)
        pd.DataFrame(self.training).drop_duplicates(subset=["timesteps"], keep="last").to_csv(directory / "training_metrics.csv", index=False)


def evaluate_policy_on_scenario(
    model: Predictor,
    scenario: str = "normal",
    episodes: int = 12,
    seed: int = 10_000,
) -> dict[str, float]:
    rewards: list[float] = []
    lengths: list[int] = []
    unmet: list[float] = []
    energy: list[float] = []
    failures = 0
    for episode in range(episodes):
        env = AquaForecastEnv()
        obs, _ = env.reset(seed=seed + episode, options={"scenario": scenario})
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += reward
            length += 1
        rewards.append(total_reward)
        lengths.append(length)
        unmet.append(float(info["total_unmet_ml"]))
        energy.append(float(info["total_energy_cost"]))
        failures += int(terminated)
        env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_unmet_ml": float(np.mean(unmet)),
        "mean_energy_cost": float(np.mean(energy)),
        "failure_rate": float(failures / episodes),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

