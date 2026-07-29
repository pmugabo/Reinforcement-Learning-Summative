"""Ten deliberately varied hyperparameter combinations per algorithm."""

from __future__ import annotations


DQN_CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.95, "buffer_size": 5_000, "batch_size": 32, "exploration_fraction": 0.30, "exploration_final_eps": 0.08, "target_update_interval": 500},
    {"learning_rate": 3e-4, "gamma": 0.97, "buffer_size": 10_000, "batch_size": 64, "exploration_fraction": 0.40, "exploration_final_eps": 0.05, "target_update_interval": 750},
    {"learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 10_000, "batch_size": 64, "exploration_fraction": 0.50, "exploration_final_eps": 0.03, "target_update_interval": 1_000},
    {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 15_000, "batch_size": 128, "exploration_fraction": 0.35, "exploration_final_eps": 0.05, "target_update_interval": 500},
    {"learning_rate": 5e-5, "gamma": 0.995, "buffer_size": 15_000, "batch_size": 128, "exploration_fraction": 0.70, "exploration_final_eps": 0.02, "target_update_interval": 1_500},
    {"learning_rate": 2e-4, "gamma": 0.90, "buffer_size": 5_000, "batch_size": 32, "exploration_fraction": 0.20, "exploration_final_eps": 0.10, "target_update_interval": 250},
    {"learning_rate": 7e-4, "gamma": 0.97, "buffer_size": 12_500, "batch_size": 64, "exploration_fraction": 0.60, "exploration_final_eps": 0.01, "target_update_interval": 1_000},
    {"learning_rate": 1e-4, "gamma": 0.999, "buffer_size": 20_000, "batch_size": 256, "exploration_fraction": 0.55, "exploration_final_eps": 0.04, "target_update_interval": 2_000},
    {"learning_rate": 3e-4, "gamma": 0.93, "buffer_size": 7_500, "batch_size": 64, "exploration_fraction": 0.25, "exploration_final_eps": 0.15, "target_update_interval": 400},
    {"learning_rate": 8e-4, "gamma": 0.995, "buffer_size": 17_500, "batch_size": 128, "exploration_fraction": 0.65, "exploration_final_eps": 0.02, "target_update_interval": 1_250},
]

REINFORCE_CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.95, "hidden_size": 64, "entropy_coef": 0.000, "batch_episodes": 4},
    {"learning_rate": 3e-4, "gamma": 0.97, "hidden_size": 64, "entropy_coef": 0.005, "batch_episodes": 4},
    {"learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 64, "entropy_coef": 0.010, "batch_episodes": 8},
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.010, "batch_episodes": 8},
    {"learning_rate": 5e-5, "gamma": 0.995, "hidden_size": 128, "entropy_coef": 0.020, "batch_episodes": 12},
    {"learning_rate": 2e-4, "gamma": 0.90, "hidden_size": 32, "entropy_coef": 0.000, "batch_episodes": 4},
    {"learning_rate": 7e-4, "gamma": 0.97, "hidden_size": 128, "entropy_coef": 0.030, "batch_episodes": 8},
    {"learning_rate": 1e-4, "gamma": 0.999, "hidden_size": 256, "entropy_coef": 0.015, "batch_episodes": 12},
    {"learning_rate": 3e-4, "gamma": 0.93, "hidden_size": 32, "entropy_coef": 0.040, "batch_episodes": 6},
    {"learning_rate": 8e-4, "gamma": 0.995, "hidden_size": 64, "entropy_coef": 0.005, "batch_episodes": 10},
]

PPO_CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.95, "n_steps": 128, "batch_size": 32, "gae_lambda": 0.90, "clip_range": 0.10, "ent_coef": 0.000},
    {"learning_rate": 3e-4, "gamma": 0.97, "n_steps": 256, "batch_size": 64, "gae_lambda": 0.95, "clip_range": 0.20, "ent_coef": 0.005},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "gae_lambda": 0.98, "clip_range": 0.20, "ent_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 128, "batch_size": 64, "gae_lambda": 0.95, "clip_range": 0.30, "ent_coef": 0.010},
    {"learning_rate": 5e-5, "gamma": 0.995, "n_steps": 512, "batch_size": 128, "gae_lambda": 0.99, "clip_range": 0.10, "ent_coef": 0.020},
    {"learning_rate": 2e-4, "gamma": 0.90, "n_steps": 64, "batch_size": 32, "gae_lambda": 0.85, "clip_range": 0.25, "ent_coef": 0.000},
    {"learning_rate": 7e-4, "gamma": 0.97, "n_steps": 256, "batch_size": 32, "gae_lambda": 0.92, "clip_range": 0.15, "ent_coef": 0.030},
    {"learning_rate": 1e-4, "gamma": 0.999, "n_steps": 512, "batch_size": 64, "gae_lambda": 0.99, "clip_range": 0.20, "ent_coef": 0.015},
    {"learning_rate": 3e-4, "gamma": 0.93, "n_steps": 128, "batch_size": 32, "gae_lambda": 0.88, "clip_range": 0.30, "ent_coef": 0.040},
    {"learning_rate": 8e-4, "gamma": 0.995, "n_steps": 256, "batch_size": 128, "gae_lambda": 0.97, "clip_range": 0.12, "ent_coef": 0.005},
]

A2C_CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.95, "n_steps": 5, "gae_lambda": 0.90, "ent_coef": 0.000, "vf_coef": 0.50, "max_grad_norm": 0.50},
    {"learning_rate": 3e-4, "gamma": 0.97, "n_steps": 10, "gae_lambda": 0.95, "ent_coef": 0.005, "vf_coef": 0.50, "max_grad_norm": 0.50},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 20, "gae_lambda": 0.98, "ent_coef": 0.010, "vf_coef": 0.60, "max_grad_norm": 0.50},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 10, "gae_lambda": 0.95, "ent_coef": 0.010, "vf_coef": 0.40, "max_grad_norm": 0.70},
    {"learning_rate": 5e-5, "gamma": 0.995, "n_steps": 40, "gae_lambda": 0.99, "ent_coef": 0.020, "vf_coef": 0.70, "max_grad_norm": 0.30},
    {"learning_rate": 2e-4, "gamma": 0.90, "n_steps": 5, "gae_lambda": 0.85, "ent_coef": 0.000, "vf_coef": 0.50, "max_grad_norm": 1.00},
    {"learning_rate": 7e-4, "gamma": 0.97, "n_steps": 20, "gae_lambda": 0.92, "ent_coef": 0.030, "vf_coef": 0.60, "max_grad_norm": 0.50},
    {"learning_rate": 1e-4, "gamma": 0.999, "n_steps": 40, "gae_lambda": 0.99, "ent_coef": 0.015, "vf_coef": 0.80, "max_grad_norm": 0.30},
    {"learning_rate": 3e-4, "gamma": 0.93, "n_steps": 10, "gae_lambda": 0.88, "ent_coef": 0.040, "vf_coef": 0.40, "max_grad_norm": 0.70},
    {"learning_rate": 8e-4, "gamma": 0.995, "n_steps": 20, "gae_lambda": 0.97, "ent_coef": 0.005, "vf_coef": 0.70, "max_grad_norm": 0.40},
]

CONFIGS = {
    "dqn": DQN_CONFIGS,
    "reinforce": REINFORCE_CONFIGS,
    "ppo": PPO_CONFIGS,
    "a2c": A2C_CONFIGS,
}
