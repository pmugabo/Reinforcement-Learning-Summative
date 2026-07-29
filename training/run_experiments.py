"""Run the required ten hyperparameter experiments for each algorithm."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from training.common import ROOT, save_json
from training.configs import CONFIGS
from training.dqn_training import train_dqn
from training.pg_training import train_a2c, train_ppo
from training.reinforce_training import train_reinforce


def run_experiments(algorithm: str, timesteps: int = 25_000, episodes: int = 250) -> pd.DataFrame:
    if algorithm not in CONFIGS:
        raise ValueError(f"Unknown algorithm {algorithm}")
    root = ROOT / "logs" / "experiments" / algorithm
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for index, config in enumerate(CONFIGS[algorithm], start=1):
        run_dir = root / f"run_{index:02d}"
        seed = 1_000 + index
        print(f"[{algorithm.upper()}] experiment {index:02d}/10 | seed={seed} | {config}", flush=True)
        if algorithm == "dqn":
            metrics = train_dqn(config, run_dir, timesteps, seed)
        elif algorithm == "ppo":
            metrics = train_ppo(config, run_dir, timesteps, seed)
        elif algorithm == "a2c":
            metrics = train_a2c(config, run_dir, timesteps, seed)
        else:
            metrics = train_reinforce(config, run_dir, episodes, seed)
        row = {"run": index, **config, **metrics}
        results.append(row)
        print(f"  mean_reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}", flush=True)

    frame = pd.DataFrame(results).sort_values("run")
    frame.to_csv(root / "summary.csv", index=False)
    best_row = frame.sort_values(["mean_reward", "failure_rate"], ascending=[False, True]).iloc[0]
    best_run = int(best_row["run"])
    best_source = root / f"run_{best_run:02d}"
    model_dir = ROOT / "models" / algorithm
    model_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "reinforce":
        shutil.copy2(best_source / "model.pt", model_dir / "best_model.pt")
    else:
        shutil.copy2(best_source / "model.zip", model_dir / "best_model.zip")
    shutil.copy2(best_source / "episodes.csv", model_dir / "best_episodes.csv")
    shutil.copy2(best_source / "training_metrics.csv", model_dir / "best_training_metrics.csv")
    best_payload = {
        "algorithm": algorithm,
        "best_run": best_run,
        "config": CONFIGS[algorithm][best_run - 1],
        "metrics": {key: float(best_row[key]) for key in ["mean_reward", "std_reward", "mean_episode_length", "mean_unmet_ml", "mean_energy_cost", "failure_rate"]},
    }
    save_json(model_dir / "best_result.json", best_payload)
    print(f"Best {algorithm.upper()} run: {best_run:02d} | mean_reward={best_row['mean_reward']:.2f}")
    return frame


def run_all(timesteps: int = 25_000, reinforce_episodes: int = 250) -> None:
    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        run_experiments(algorithm, timesteps=timesteps, episodes=reinforce_episodes)
