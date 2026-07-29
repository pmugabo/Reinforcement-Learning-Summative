"""Mission-based water-allocation environment for AquaForecast.

The environment represents a daily decision-support cycle for a community water
system in Kimironko Sector, Kigali.  An action is a recommendation to operators;
it is not a command sent to physical infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ActionProfile:
    name: str
    description: str
    production_multiplier: float = 1.0
    reserve_release: float = 0.0
    backup_supply: float = 0.0
    conservation_days: int = 0
    leak_reduction: float = 0.0
    maintenance: bool = False
    operating_cost: float = 0.0


class AquaForecastEnv(gym.Env[np.ndarray, int]):
    """Stochastic 90-day water-demand planning environment.

    Observation values are normalized to [0, 1].  The action space is discrete
    so DQN, REINFORCE, PPO and A2C can be compared on exactly the same task.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    CAPACITY_ML = 12.0
    MAX_DAILY_DEMAND_ML = 0.95
    MAX_DAILY_PRODUCTION_ML = 0.78
    EPISODE_DAYS = 90

    ACTIONS: tuple[ActionProfile, ...] = (
        ActionProfile("Maintain forecast plan", "Produce the forecast quantity and monitor storage."),
        ActionProfile("Reduce production 20%", "Save energy when demand and shortage risk are low.", production_multiplier=0.80),
        ActionProfile("Increase production 20%", "Build storage before a moderate demand peak.", production_multiplier=1.20, operating_cost=0.05),
        ActionProfile("Increase production 40%", "Respond to a high-demand or low-storage warning.", production_multiplier=1.40, operating_cost=0.10),
        ActionProfile("Release strategic reserve", "Make a controlled 0.16 ML emergency reserve release.", reserve_release=0.16, operating_cost=0.24),
        ActionProfile("Activate backup supply", "Use a borehole/tanker-equivalent 0.13 ML supply source.", backup_supply=0.13, operating_cost=0.55),
        ActionProfile("Issue conservation advisory", "Reduce demand for five days through public guidance.", conservation_days=5, operating_cost=0.16),
        ActionProfile("Dispatch leak-response crew", "Inspect pressure/leaks and repair the highest-risk segment.", leak_reduction=0.48, operating_cost=0.16),
        ActionProfile("Combined drought response", "Increase production and issue a seven-day conservation advisory.", production_multiplier=1.20, conservation_days=7, operating_cost=0.35),
        ActionProfile("Schedule preventive maintenance", "Temporarily reduce throughput to restore plant reliability.", maintenance=True, operating_cost=0.12),
    )

    OBSERVATION_NAMES = (
        "reservoir_level",
        "forecast_demand",
        "previous_demand",
        "rainfall_forecast",
        "temperature",
        "humidity",
        "dry_season_index",
        "leakage_risk",
        "treatment_health",
        "energy_tariff",
        "forecast_uncertainty",
        "conservation_active",
        "maintenance_benefit",
        "strategic_reserve",
        "day_of_year_sin",
        "day_of_year_cos",
    )

    def __init__(self, render_mode: str | None = None, episode_days: int = EPISODE_DAYS):
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.episode_days = episode_days
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.OBSERVATION_NAMES), dtype=np.float32),
            high=np.ones(len(self.OBSERVATION_NAMES), dtype=np.float32),
            dtype=np.float32,
        )
        self._renderer = None
        self._scenario = "normal"
        self._last_action = 0
        self._last_reward = 0.0
        self._last_info: dict[str, Any] = {}

    @classmethod
    def action_catalog(cls) -> list[dict[str, Any]]:
        return [
            {"id": idx, "name": action.name, "description": action.description}
            for idx, action in enumerate(cls.ACTIONS)
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._scenario = str(options.get("scenario", "normal"))
        allowed = {"normal", "drought", "demand_surge", "high_leakage", "plant_outage", "mixed"}
        if self._scenario not in allowed:
            raise ValueError(f"Unknown scenario {self._scenario!r}; choose one of {sorted(allowed)}")

        self.day = 0
        self.day_of_year = int(self.np_random.integers(0, 365))
        self.reservoir_ml = float(self.np_random.uniform(5.4, 9.6))
        self.previous_demand_ml = float(self.np_random.uniform(0.40, 0.62))
        self.treatment_health = float(self.np_random.uniform(0.82, 1.0))
        self.pipe_health = float(self.np_random.uniform(0.72, 0.96))
        self.conservation_days_left = 0
        self.maintenance_benefit_days = 0
        self.strategic_reserve_ml = float(self.np_random.uniform(1.2, 2.5))
        self.critical_shortage_days = 0
        self.total_unmet_ml = 0.0
        self.total_energy_cost = 0.0
        self.total_overflow_ml = 0.0
        self._last_action = 0
        self._last_reward = 0.0

        if self._scenario in {"drought", "mixed"}:
            self.reservoir_ml *= 0.68
            self.strategic_reserve_ml *= 0.72
        if self._scenario in {"high_leakage", "mixed"}:
            self.pipe_health = float(self.np_random.uniform(0.48, 0.62))
        if self._scenario == "plant_outage":
            self.treatment_health = float(self.np_random.uniform(0.50, 0.62))

        self._sample_exogenous_inputs()
        observation = self._get_observation()
        info = self._build_info(0.0, {}, terminated=False, truncated=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _sample_exogenous_inputs(self) -> None:
        seasonal = 0.5 + 0.5 * np.sin(2.0 * np.pi * (self.day_of_year - 170) / 365.0)
        self.dry_season_index = float(np.clip(seasonal, 0.0, 1.0))
        drought_boost = 0.28 if self._scenario in {"drought", "mixed"} else 0.0
        rain_mean = 9.0 * (1.0 - self.dry_season_index) * (1.0 - drought_boost)
        self.rainfall_mm = float(np.clip(self.np_random.gamma(1.6, max(0.35, rain_mean / 1.6)), 0.0, 25.0))
        self.temperature_c = float(np.clip(22.5 + 5.3 * self.dry_season_index + self.np_random.normal(0, 1.1), 17.0, 35.0))
        self.humidity = float(np.clip(0.78 - 0.30 * self.dry_season_index + self.np_random.normal(0, 0.05), 0.25, 0.95))
        peak = 0.07 if (self.day_of_year % 7) in {5, 6} else 0.0
        trend = 0.0007 * self.day
        surge = 0.15 if self._scenario in {"demand_surge", "mixed"} else 0.0
        conservation = 0.11 if self.conservation_days_left > 0 else 0.0
        base = 0.42 + 0.16 * self.dry_season_index + 0.010 * (self.temperature_c - 24.0) + peak + trend + surge
        self.actual_demand_ml = float(np.clip(base * (1.0 - conservation) + self.np_random.normal(0, 0.025), 0.25, self.MAX_DAILY_DEMAND_ML))
        scenario_uncertainty = 0.06 if self._scenario == "mixed" else 0.0
        self.forecast_uncertainty = float(np.clip(0.06 + 0.10 * self.dry_season_index + scenario_uncertainty, 0.04, 0.25))
        error = self.np_random.normal(0, self.forecast_uncertainty * self.actual_demand_ml)
        self.forecast_demand_ml = float(np.clip(self.actual_demand_ml + error, 0.20, self.MAX_DAILY_DEMAND_ML))
        tariff_peak = 0.30 if 18 <= (self.day % 24) <= 21 else 0.0
        self.energy_tariff = float(np.clip(0.45 + tariff_peak + 0.20 * self.np_random.random(), 0.40, 1.0))
        self.leakage_rate = float(np.clip(0.035 + 0.16 * (1.0 - self.pipe_health), 0.025, 0.18))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action = int(action)
        profile = self.ACTIONS[action]
        self._last_action = action

        demand = self.actual_demand_ml
        conservation_already_active = self.conservation_days_left > 0
        if profile.conservation_days:
            self.conservation_days_left = max(self.conservation_days_left, profile.conservation_days)
            demand *= 0.94

        if profile.maintenance:
            capacity_factor = 0.48
            self.treatment_health = min(1.0, self.treatment_health + 0.15)
            self.maintenance_benefit_days = 14
        else:
            capacity_factor = self.treatment_health

        reservoir_fraction = self.reservoir_ml / self.CAPACITY_ML
        storage_bias = np.clip((0.58 - reservoir_fraction) * 0.30, -0.08, 0.14)
        planned_production = (self.forecast_demand_ml + storage_bias) * profile.production_multiplier
        production_ml = float(np.clip(planned_production, 0.0, self.MAX_DAILY_PRODUCTION_ML * capacity_factor))
        rainfall_capture_ml = float(min(0.10, self.rainfall_mm * 0.004))
        reserve_release_ml = min(profile.reserve_release, self.strategic_reserve_ml)
        self.strategic_reserve_ml -= reserve_release_ml
        emergency_ml = reserve_release_ml + profile.backup_supply

        leak_reduction = profile.leak_reduction
        leak_volume_ml = demand * self.leakage_rate * (1.0 - leak_reduction)
        if profile.leak_reduction > 0:
            self.pipe_health = min(1.0, self.pipe_health + 0.09)

        available_ml = self.reservoir_ml + production_ml + rainfall_capture_ml + emergency_ml
        distributable_ml = max(0.0, available_ml - leak_volume_ml)
        served_ml = min(demand, distributable_ml)
        unmet_ml = max(0.0, demand - served_ml)
        remaining_ml = max(0.0, distributable_ml - served_ml)
        overflow_ml = max(0.0, remaining_ml - self.CAPACITY_ML)
        self.reservoir_ml = min(self.CAPACITY_ML, remaining_ml)

        service_ratio = served_ml / max(demand, 1e-6)
        unmet_ratio = unmet_ml / max(demand, 1e-6)
        energy_cost = production_ml * self.energy_tariff
        level_fraction = self.reservoir_ml / self.CAPACITY_ML

        reward_components = {
            "service": 5.0 * service_ratio,
            "shortage": -15.0 * unmet_ratio**2 - (4.0 if unmet_ratio > 0.10 else 0.0),
            "storage_balance": -2.0 * abs(level_fraction - 0.55),
            "energy": -1.25 * energy_cost,
            "overflow": -6.0 * (overflow_ml / self.CAPACITY_ML),
            "operational_action": -profile.operating_cost - (0.38 if conservation_already_active and profile.conservation_days else 0.0),
            "resilience_bonus": 0.45 if 0.28 <= level_fraction <= 0.82 and service_ratio >= 0.985 else 0.0,
        }
        reward = float(sum(reward_components.values()))

        self.total_unmet_ml += unmet_ml
        self.total_energy_cost += energy_cost
        self.total_overflow_ml += overflow_ml
        self.critical_shortage_days = self.critical_shortage_days + 1 if unmet_ratio > 0.30 else 0
        terminated = self.critical_shortage_days >= 3

        self.previous_demand_ml = demand
        self.treatment_health = float(np.clip(self.treatment_health - self.np_random.uniform(0.001, 0.006), 0.45, 1.0))
        if self.maintenance_benefit_days > 0:
            self.treatment_health = min(1.0, self.treatment_health + 0.0025)
            self.maintenance_benefit_days -= 1
        self.pipe_health = float(np.clip(self.pipe_health - self.np_random.uniform(0.0005, 0.003), 0.35, 1.0))
        self.conservation_days_left = max(0, self.conservation_days_left - 1)

        self.day += 1
        self.day_of_year = (self.day_of_year + 1) % 365
        truncated = self.day >= self.episode_days
        if not (terminated or truncated):
            self._sample_exogenous_inputs()

        self._last_reward = reward
        info = self._build_info(reward, reward_components, terminated, truncated)
        self._last_info = info
        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        angle = 2.0 * np.pi * self.day_of_year / 365.0
        values = np.array(
            [
                self.reservoir_ml / self.CAPACITY_ML,
                self.forecast_demand_ml / self.MAX_DAILY_DEMAND_ML,
                self.previous_demand_ml / self.MAX_DAILY_DEMAND_ML,
                self.rainfall_mm / 25.0,
                (self.temperature_c - 17.0) / 18.0,
                self.humidity,
                self.dry_season_index,
                self.leakage_rate / 0.18,
                self.treatment_health,
                (self.energy_tariff - 0.40) / 0.60,
                self.forecast_uncertainty / 0.25,
                self.conservation_days_left / 7.0,
                self.maintenance_benefit_days / 14.0,
                self.strategic_reserve_ml / 2.5,
                (np.sin(angle) + 1.0) / 2.0,
                (np.cos(angle) + 1.0) / 2.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    def _build_info(
        self,
        reward: float,
        reward_components: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "day": self.day,
            "scenario": self._scenario,
            "action_id": self._last_action,
            "action_name": self.ACTIONS[self._last_action].name,
            "reward": reward,
            "reward_components": reward_components,
            "reservoir_ml": self.reservoir_ml,
            "forecast_demand_ml": self.forecast_demand_ml,
            "actual_demand_ml": self.actual_demand_ml,
            "rainfall_mm": self.rainfall_mm,
            "temperature_c": self.temperature_c,
            "leakage_rate": self.leakage_rate,
            "treatment_health": self.treatment_health,
            "energy_tariff": self.energy_tariff,
            "total_unmet_ml": self.total_unmet_ml,
            "total_energy_cost": self.total_energy_cost,
            "total_overflow_ml": self.total_overflow_ml,
            "strategic_reserve_ml": self.strategic_reserve_ml,
            "terminated": terminated,
            "truncated": truncated,
        }

    def api_state(self) -> dict[str, Any]:
        """Return a frontend-ready JSON-serializable state payload."""
        return {
            "mission": "AquaForecast",
            "location": "Kimironko Sector, Gasabo District, Kigali, Rwanda",
            "timestamp": {"simulation_day": self.day, "day_of_year": self.day_of_year},
            "state": {
                "reservoir_ml": round(self.reservoir_ml, 4),
                "reservoir_percent": round(100 * self.reservoir_ml / self.CAPACITY_ML, 2),
                "forecast_demand_ml": round(self.forecast_demand_ml, 4),
                "actual_demand_ml": round(self.actual_demand_ml, 4),
                "rainfall_mm": round(self.rainfall_mm, 2),
                "temperature_c": round(self.temperature_c, 2),
                "leakage_percent": round(100 * self.leakage_rate, 2),
                "treatment_health_percent": round(100 * self.treatment_health, 2),
                "strategic_reserve_ml": round(self.strategic_reserve_ml, 3),
            },
            "recommendation": {
                "action_id": self._last_action,
                "action": self.ACTIONS[self._last_action].name,
                "description": self.ACTIONS[self._last_action].description,
                "confidence_note": "Simulation recommendation; human operator approval required.",
            },
            "reward": round(self._last_reward, 4),
        }

    def render(self) -> np.ndarray | None:
        from environment.rendering import AquaForecastRenderer, draw_rgb_frame

        if self.render_mode == "rgb_array":
            return draw_rgb_frame(self)
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = AquaForecastRenderer(self)
            self._renderer.draw(self)
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
