"""Generate every visualization required by the rubric."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from environment.custom_env import AquaForecastEnv
from training.common import ROOT


ALGORITHMS = ("dqn", "reinforce", "ppo", "a2c")
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2563EB", "reinforce": "#F97316", "ppo": "#10B981", "a2c": "#8B5CF6"}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "axes.titleweight": "bold"})


def _load_episodes(algorithm: str) -> pd.DataFrame:
    path = ROOT / "models" / algorithm / "best_episodes.csv"
    frame = pd.read_csv(path)
    frame["episode"] = np.arange(1, len(frame) + 1)
    return frame


def cumulative_rewards_plot(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        frame = _load_episodes(algorithm)
        cumulative = frame["reward"].cumsum()
        ax.plot(frame["episode"], cumulative, color=COLORS[algorithm], linewidth=2)
        ax.fill_between(frame["episode"], cumulative, alpha=0.12, color=COLORS[algorithm])
        ax.set_title(LABELS[algorithm])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
    fig.suptitle("Cumulative Reward During Training — Best Run per Algorithm", fontsize=14, fontweight="bold")
    fig.savefig(output / "cumulative_rewards_all_methods.png", bbox_inches="tight")
    plt.close(fig)


def learning_curves_plot(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for algorithm in ALGORITHMS:
        frame = _load_episodes(algorithm)
        rolling = frame["reward"].rolling(10, min_periods=1).mean()
        ax.plot(frame["episode"], rolling, label=LABELS[algorithm], color=COLORS[algorithm], linewidth=2)
    ax.set_title("Training Learning Curves (10-Episode Rolling Mean)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.legend(ncol=4, frameon=True)
    fig.savefig(output / "learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def stability_plot(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        frame = pd.read_csv(ROOT / "models" / algorithm / "best_training_metrics.csv")
        if algorithm == "dqn":
            metric = "loss"
            label = "TD objective loss"
        else:
            metric = "entropy"
            label = "Policy entropy"
        clean = frame.dropna(subset=[metric])
        if clean.empty:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
        else:
            smooth = clean[metric].rolling(8, min_periods=1).mean()
            ax.plot(clean["timesteps"], smooth, color=COLORS[algorithm], linewidth=2)
        ax.set_title(f"{LABELS[algorithm]} — {label}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(label)
    fig.suptitle("Training Stability: DQN Objective and Policy-Gradient Entropy", fontsize=14, fontweight="bold")
    fig.savefig(output / "training_stability.png", bbox_inches="tight")
    plt.close(fig)


def _convergence_episode(frame: pd.DataFrame) -> int:
    rewards = frame["reward"].to_numpy(float)
    rolling = pd.Series(rewards).rolling(20, min_periods=20).mean().to_numpy()
    valid = rolling[np.isfinite(rolling)]
    if valid.size == 0:
        return len(frame)
    peak = float(np.max(valid))
    threshold = peak - 0.02 * max(1.0, abs(peak))
    patience = 10
    required_hits = 8
    for index in range(19, len(rolling) - patience + 1):
        window = rolling[index : index + patience]
        if np.isfinite(window).all() and int(np.sum(window >= threshold)) >= required_hits:
            return index + 1
    return len(frame)


def convergence_plot(output: Path) -> None:
    values = [_convergence_episode(_load_episodes(algorithm)) for algorithm in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    bars = ax.bar([LABELS[a] for a in ALGORITHMS], values, color=[COLORS[a] for a in ALGORITHMS])
    ax.bar_label(bars, padding=3, fmt="%d episodes")
    ax.set_title("Episodes to Stable Performance")
    ax.set_ylabel("Convergence episode (lower is faster)")
    ax.set_ylim(0, max(values) * 1.20)
    fig.savefig(output / "episodes_to_converge.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({"algorithm": ALGORITHMS, "episodes_to_converge": values}).to_csv(output / "convergence_summary.csv", index=False)


def generalization_plot(output: Path) -> None:
    path = ROOT / "logs" / "evaluation" / "generalization_results.csv"
    frame = pd.read_csv(path)
    frame["algorithm_label"] = frame["algorithm"].map(LABELS)
    frame["scenario_label"] = frame["scenario"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    sns.barplot(data=frame, x="scenario_label", y="mean_reward", hue="algorithm_label", palette=[COLORS[a] for a in ALGORITHMS], ax=ax)
    ax.set_title("Generalization to Unseen Water-System Conditions")
    ax.set_xlabel("Evaluation scenario")
    ax.set_ylabel("Mean episode reward (20 seeds)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Algorithm", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.savefig(output / "generalization_tests.png", bbox_inches="tight")
    plt.close(fig)


def environment_snapshot(output: Path) -> None:
    env = AquaForecastEnv(render_mode="rgb_array")
    env.reset(seed=2026, options={"scenario": "drought"})
    for action in (2, 7, 6, 8):
        env.step(action)
    frame = env.render()
    Image.fromarray(frame).save(output / "environment_visual.png")
    env.close()


def generate_all() -> None:
    _style()
    output = ROOT / "assets"
    output.mkdir(parents=True, exist_ok=True)
    cumulative_rewards_plot(output)
    learning_curves_plot(output)
    stability_plot(output)
    convergence_plot(output)
    generalization_plot(output)
    environment_snapshot(output)
    print(f"Saved report visuals to {output}")
