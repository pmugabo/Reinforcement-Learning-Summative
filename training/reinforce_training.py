"""A compact Monte-Carlo REINFORCE implementation using PyTorch.

Stable-Baselines3 provides DQN, PPO and A2C but does not ship REINFORCE.  This
module keeps the same Gymnasium environment and SB3-style predict/save workflow
while implementing the required algorithm directly in PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from environment.custom_env import AquaForecastEnv
from training.common import evaluate_policy_on_scenario, save_json, set_global_seed


class PolicyNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class ReinforcePolicy:
    def __init__(self, network: PolicyNetwork):
        self.network = network.eval()

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            distribution = Categorical(logits=self.network(tensor))
            action = torch.argmax(distribution.probs, dim=-1) if deterministic else distribution.sample()
        return int(action.item()), None

    def save(self, path: Path, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.network.state_dict(), "config": config}, path)

    @classmethod
    def load(cls, path: Path) -> "ReinforcePolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        network = PolicyNetwork(
            len(AquaForecastEnv.OBSERVATION_NAMES),
            len(AquaForecastEnv.ACTIONS),
            int(config["hidden_size"]),
        )
        network.load_state_dict(checkpoint["state_dict"])
        return cls(network)


def _discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        values.append(running)
    returns = torch.tensor(list(reversed(values)), dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def train_reinforce(
    config: dict[str, Any],
    run_dir: Path,
    total_episodes: int,
    seed: int,
) -> dict[str, float]:
    set_global_seed(seed)
    env = AquaForecastEnv()
    network = PolicyNetwork(
        len(AquaForecastEnv.OBSERVATION_NAMES),
        len(AquaForecastEnv.ACTIONS),
        int(config["hidden_size"]),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=float(config["learning_rate"]))
    batch_losses: list[torch.Tensor] = []
    rows: list[dict[str, float]] = []
    batch_size = int(config["batch_episodes"])

    for episode in range(total_episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        while not (terminated or truncated):
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            distribution = Categorical(logits=network(tensor))
            action = distribution.sample()
            log_probs.append(distribution.log_prob(action).squeeze(0))
            entropies.append(distribution.entropy().squeeze(0))
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(reward)

        returns = _discounted_returns(rewards, float(config["gamma"]))
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        episode_loss = -(log_prob_tensor * returns).mean() - float(config["entropy_coef"]) * entropy_tensor.mean()
        batch_losses.append(episode_loss)

        update_loss = np.nan
        if len(batch_losses) >= batch_size or episode == total_episodes - 1:
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            update_loss = float(loss.detach().item())
            batch_losses.clear()

        rows.append(
            {
                "episode": episode + 1,
                "timesteps": float(sum(int(row["length"]) for row in rows) + len(rewards)),
                "reward": float(sum(rewards)),
                "length": float(len(rewards)),
                "entropy": float(torch.stack(entropies).mean().detach().item()),
                "loss": update_loss,
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    policy = ReinforcePolicy(network)
    policy.save(run_dir / "model.pt", config)
    frame = pd.DataFrame(rows)
    frame[["episode", "timesteps", "reward", "length"]].to_csv(run_dir / "episodes.csv", index=False)
    frame[["timesteps", "entropy", "loss"]].to_csv(run_dir / "training_metrics.csv", index=False)
    metrics = evaluate_policy_on_scenario(policy, seed=20_000 + seed)
    save_json(run_dir / "result.json", {"algorithm": "reinforce", "seed": seed, "config": config, "metrics": metrics})
    env.close()
    return metrics

