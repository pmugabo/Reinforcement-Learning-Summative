"""Command-line entry point for AquaForecast RL Summative."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from stable_baselines3.common.env_checker import check_env

from analysis.evaluate import evaluate_all, load_model
from analysis.generate_plots import generate_all
from environment.custom_env import AquaForecastEnv
from training.run_experiments import run_all, run_experiments


def command_check(_: argparse.Namespace) -> None:
    env = AquaForecastEnv()
    check_env(env, warn=True, skip_render_check=True)
    obs, info = env.reset(seed=42)
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, info = env.reset()
    print("Environment check passed: Gymnasium/SB3 API, bounds, reset and terminal behavior are valid.")
    print(f"Observation shape: {obs.shape}; actions: {env.action_space.n}")
    env.close()


def command_experiments(args: argparse.Namespace) -> None:
    if args.algorithm == "all":
        run_all(timesteps=args.timesteps, reinforce_episodes=args.episodes)
    else:
        run_experiments(args.algorithm, timesteps=args.timesteps, episodes=args.episodes)


def command_play(args: argparse.Namespace) -> None:
    model = load_model(args.algorithm)
    render_mode = None if args.no_render else "human"
    env = AquaForecastEnv(render_mode=render_mode)
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode, options={"scenario": args.scenario})
        terminated = truncated = False
        total_reward = 0.0
        print(f"\nEpisode {episode + 1} | algorithm={args.algorithm.upper()} | scenario={args.scenario}")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += reward
            print(
                f"day={info['day']:02d} action={info['action_name']:<31} "
                f"storage={info['reservoir_ml']:5.2f}ML demand={info['actual_demand_ml']:.3f}ML "
                f"reward={reward:+6.2f}",
                flush=True,
            )
            if not args.no_render:
                time.sleep(args.delay)
        print(f"Episode complete | total reward={total_reward:.2f} | unmet={info['total_unmet_ml']:.3f} ML")
    env.close()


def command_api_state(args: argparse.Namespace) -> None:
    env = AquaForecastEnv()
    obs, _ = env.reset(seed=args.seed, options={"scenario": args.scenario})
    if args.algorithm:
        model = load_model(args.algorithm)
        action, _ = model.predict(obs, deterministic=True)
        env.step(int(np.asarray(action).item()))
    print(json.dumps(env.api_state(), indent=2))
    env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AquaForecast mission-based reinforcement-learning project")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Validate the custom Gymnasium environment")
    check.set_defaults(func=command_check)

    experiments = commands.add_parser("experiments", help="Run the ten configurations for one or all algorithms")
    experiments.add_argument("--algorithm", choices=["all", "dqn", "reinforce", "ppo", "a2c"], default="all")
    experiments.add_argument("--timesteps", type=int, default=25_000, help="Timesteps per SB3 run")
    experiments.add_argument("--episodes", type=int, default=250, help="Episodes per REINFORCE run")
    experiments.set_defaults(func=command_experiments)

    evaluate = commands.add_parser("evaluate", help="Test best models on six unseen scenarios")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.set_defaults(func=lambda args: evaluate_all(args.episodes))

    plots = commands.add_parser("plots", help="Generate all rubric-required visualizations")
    plots.set_defaults(func=lambda _: generate_all())

    play = commands.add_parser("play", help="Run a trained agent with GUI and verbose terminal output")
    play.add_argument("--algorithm", choices=["dqn", "reinforce", "ppo", "a2c"], default="ppo")
    play.add_argument("--scenario", choices=["normal", "drought", "demand_surge", "high_leakage", "plant_outage", "mixed"], default="high_leakage")
    play.add_argument("--episodes", type=int, default=1)
    play.add_argument("--seed", type=int, default=2026)
    play.add_argument("--delay", type=float, default=0.08)
    play.add_argument("--no-render", action="store_true")
    play.set_defaults(func=command_play)

    api_state = commands.add_parser("api-state", help="Serialize one decision state as frontend-ready JSON")
    api_state.add_argument("--algorithm", choices=["dqn", "reinforce", "ppo", "a2c"])
    api_state.add_argument("--scenario", default="normal")
    api_state.add_argument("--seed", type=int, default=42)
    api_state.set_defaults(func=command_api_state)
    return parser


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("check")
    arguments = build_parser().parse_args()
    arguments.func(arguments)
