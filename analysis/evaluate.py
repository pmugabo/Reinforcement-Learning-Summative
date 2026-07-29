"""Evaluate best models on normal and unseen generalization scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import A2C, DQN, PPO

from training.common import ROOT, evaluate_policy_on_scenario
from training.reinforce_training import ReinforcePolicy


SCENARIOS = ("normal", "drought", "demand_surge", "high_leakage", "plant_outage", "mixed")


def load_model(algorithm: str):
    model_dir = ROOT / "models" / algorithm
    if algorithm == "dqn":
        return DQN.load(model_dir / "best_model.zip", device="cpu")
    if algorithm == "ppo":
        return PPO.load(model_dir / "best_model.zip", device="cpu")
    if algorithm == "a2c":
        return A2C.load(model_dir / "best_model.zip", device="cpu")
    if algorithm == "reinforce":
        return ReinforcePolicy.load(model_dir / "best_model.pt")
    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_all(episodes: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        model = load_model(algorithm)
        for index, scenario in enumerate(SCENARIOS):
            metrics = evaluate_policy_on_scenario(
                model,
                scenario=scenario,
                episodes=episodes,
                seed=50_000 + 1_000 * index,
            )
            rows.append({"algorithm": algorithm, "scenario": scenario, **metrics})
            print(f"{algorithm.upper():10s} | {scenario:13s} | reward={metrics['mean_reward']:8.2f} | unmet={metrics['mean_unmet_ml']:.3f} ML")
    frame = pd.DataFrame(rows)
    output = ROOT / "logs" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "generalization_results.csv", index=False)
    return frame

