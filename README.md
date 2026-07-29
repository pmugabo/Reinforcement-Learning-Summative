# AquaForecast Mission-Based Reinforcement Learning

This project compares DQN, REINFORCE, PPO, and A2C on one custom Gymnasium environment representing daily water-operation recommendations for Kimironko Sector, Gasabo District, Kigali. The agent balances reliable water service, reservoir resilience, treatment capacity, leakage, energy cost, weather, and uncertain demand forecasts.

## Mission and objective

AquaForecast is a water-demand forecasting and decision-support platform. In this simulation, the RL agent does **not** directly operate public infrastructure. It recommends one daily operational strategy to a human water-system operator. The objective is to meet demand while avoiding shortages and overflow, maintaining strategic storage, and limiting energy and emergency-action costs.

The environment is intentionally non-generic: demand changes with temperature, rainfall, dry-season intensity, weekends, conservation campaigns, forecast uncertainty, leakage risk, plant health, and generalization scenarios such as drought and plant outage.

## Project structure

```text
project_root/
├── pyproject.toml
├── uv.lock                    # created/updated by uv sync
├── README.md
├── main.py
├── play.py
├── environment/
│   ├── __init__.py
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── __init__.py
│   ├── configs.py
│   ├── common.py
│   ├── dqn_training.py
│   ├── pg_training.py
│   ├── reinforce_training.py
│   └── run_experiments.py
├── analysis/
│   ├── evaluate.py
│   └── generate_plots.py
├── models/{dqn,reinforce,ppo,a2c}/
├── logs/
├── assets/
└── tests/
```

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A normal desktop display for the OpenGL GUI
- CPU is sufficient; a GPU is optional

## Install and verify

After cloning the repository, run only uv commands:

```bash
uv sync
uv run main.py check
uv run pytest
```

`uv sync` creates the project environment and lock file automatically. No manual virtual-environment creation or `pip install` command is required.

## Run the required 40 experiments

The following command runs ten different configurations for each of the four algorithms:

```bash
uv run main.py experiments --algorithm all --timesteps 25000 --episodes 250
```

For a quick smoke test before the full run:

```bash
uv run main.py experiments --algorithm dqn --timesteps 1500
```

Each run saves its configuration, episode rewards, episode lengths, optimization metrics, evaluation results, and trained model. The highest mean evaluation reward determines the best model for each algorithm.

> Stable-Baselines3 includes DQN, PPO, and A2C but does not provide REINFORCE. The required REINFORCE model is therefore implemented directly in PyTorch while using the identical Gymnasium environment, seeds, evaluation protocol, model interface, and logging format.

## Evaluate generalization and create figures

```bash
uv run main.py evaluate --episodes 20
uv run main.py plots
```

Evaluation covers normal conditions plus unseen drought, demand surge, high leakage, plant outage, and mixed-stress states. Generated figures include cumulative reward subplots, smoothed learning curves, DQN objective loss, policy entropy, convergence episodes, generalization results, and the environment visual.

After training and evaluation, rebuild the final evidence-based report from the generated CSV files and figures:

```bash
uv run report/build_report_draft.py
```

The completed report is written to `report/AquaForecast_RL_Summative_Final.docx`. Export that file to PDF after adding the GitHub and video URLs on its first page.

## Run the best agent for the video

PPO is the demonstration model because it achieved the highest six-scenario average reward and lowest average failure rate. High leakage is used because it is a meaningful stress test on which PPO performed reliably.

```bash
uv run play.py --algorithm ppo --scenario high_leakage --episodes 1 --delay 0.08
```

The terminal prints the day, selected action, storage, demand, and reward while the OpenGL GUI shows the reservoir, system condition, and current recommendation.

## Frontend/API integration example

```bash
uv run main.py api-state --algorithm ppo --scenario drought
```

This serializes the system state and recommendation as JSON suitable for a web or mobile frontend. A real deployment must retain human approval and safety constraints.

## Reproducibility

Training and evaluation use explicit seeds. Exact neural-network results can still differ slightly by processor, operating system, and library build. Do not edit the experiment tables manually; use the generated `logs/experiments/*/summary.csv` files as the source of truth for the report.
