from __future__ import annotations

from carla_gym.envs.carla_env import CarlaEnv
from carla_gym.envs.task_config import TaskConfig


class DynamicEnv(CarlaEnv):
    def __init__(
        self,
        host: str = "localhost",
        gpu_id: int = 0,
        no_rendering: bool = False,
        nullrhi: bool = False,
        task_config: TaskConfig | None = None,
        obs_configs: dict | None = None,
        reward_configs: dict | None = None,
        terminal_configs: dict | None = None,
        render_mode: str = "rgb_array",
        **overrides_task_config: dict,
    ):
        # Set default obs_configs if not provided
        if obs_configs is None:
            obs_configs = self._get_default_obs_configs()

        if task_config is None:
            task_config = TaskConfig.sample(n=1)

        for k, v in overrides_task_config.items():
            if hasattr(task_config, k):
                setattr(task_config, k, v)
            else:
                raise ValueError(f"TaskConfig has no attribute {k}")

        super().__init__(
            map_name=task_config.map_name,
            host=host,
            seed=task_config.seed,
            gpu_id=gpu_id,
            no_rendering=no_rendering,
            nullrhi=nullrhi,
            obs_configs=obs_configs,
            reward_configs=reward_configs or self._get_default_reward_configs(),
            terminal_configs=terminal_configs or self._get_default_terminal_configs(),
            task_config=task_config,
            render_mode=render_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset environment with dynamic NPC counts if ranges are specified."""
        # Handle dynamic NPC counts
        self._update_npc_counts()

        # Call parent reset
        return super().reset(seed=seed, options=options)

    def _update_npc_counts(self) -> None:
        """Update NPC counts in current task if ranges are specified."""
        if not hasattr(self, "_task") or self._task is None:
            return

        # Update NPC vehicles count
        if isinstance(self._task_config.num_npc_vehicles, list) and len(self._task_config.num_npc_vehicles) == 2:
            min_vehicles, max_vehicles = self._task_config.num_npc_vehicles
            self._task["num_npc_vehicles"] = self._rng.integers(min_vehicles, max_vehicles + 1)

        # Update NPC walkers count
        if isinstance(self._task_config.num_npc_walkers, list) and len(self._task_config.num_npc_walkers) == 2:
            min_walkers, max_walkers = self._task_config.num_npc_walkers
            self._task["num_npc_walkers"] = self._rng.integers(min_walkers, max_walkers + 1)

    @staticmethod
    def _get_default_obs_configs() -> dict:
        """Get default observation configurations based on birdview_no_scale.yaml."""
        return {
            "hero": {
                "birdview": {
                    "module": "birdview.chauffeurnet:Birdview",
                    "width_in_pixels": 256,
                    "pixels_ev_to_bottom": 64,
                    "pixels_per_meter": 4.0,
                    "history_idx": [-16, -11, -6, -1],
                    "scale_bbox": True,
                    "scale_mask_col": 1.1,
                },
                "speed": {
                    "module": "actor_state.speed:ObsManager",
                },
                "control": {
                    "module": "actor_state.control:ObsManager",
                },
                "velocity": {
                    "module": "actor_state.velocity:ObsManager",
                },
                "front_rgb": {
                    "module": "camera.rgb:CameraRGB",
                    "type": "sensor.camera.rgb",
                    "x": -1.5,
                    "y": 0.0,
                    "z": 2.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "width": 512,
                    "height": 256,
                    "fov": 110,
                    "id": "front_rgb",
                },
            }
        }

    @staticmethod
    def _get_default_reward_configs() -> dict:
        return {
            "hero": {
                "entry_point": "reward.valeo_action:ValeoAction",
                "kwargs": {},
            }
        }

    @staticmethod
    def _get_default_terminal_configs() -> dict:
        return {
            "hero": {
                "entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx",
                "kwargs": {
                    "timeout_steps": 3000,
                },
            }
        }
