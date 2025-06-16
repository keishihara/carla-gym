from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from gymnasium import spaces

from carla_gym.engine.observation.base import BaseObservation


class ObsManagerHandler:
    """Manage ObsManager instances grouped by ego vehicle.

    The class dynamically instantiates observation managers based on the
    provided configuration and exposes a unified interface to:
      * fetch observations
      * reset / (re)attach ego vehicles
      * clean up resources
    """

    def __init__(self, obs_configs: Mapping[str, Mapping[str, dict[str, Any]]]) -> None:
        """Initialize the handler.

        Args:
            obs_configs: Nested mapping of vehicle id → observation id →
                observation configuration. Each configuration must contain
                a 'module' key that points to an importable Python module
                under `carla_gym.engine.observation`.
        """
        self._obs_managers: dict[str, dict[str, BaseObservation]] = {}
        self._obs_configs: Mapping[str, Mapping[str, dict[str, Any]]] = obs_configs
        self._init_obs_managers()

    def get_observation(self, timestamp: float | int | None = None) -> dict[str, dict[str, Any]]:
        """Collect observations from all managed `ObsManager`s.

        Args:
            timestamp: Unused but kept for API compatibility.

        Returns:
            Nested dict of vehicle id → observation id → observation data.
        """
        expected_frame = None
        if isinstance(timestamp, dict) and "frame" in timestamp:
            expected_frame = int(timestamp["frame"])

        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            obs_dict[ev_id] = {}
            for obs_id, om in om_dict.items():
                om.set_expected_frame(expected_frame)
                obs_dict[ev_id][obs_id] = om.get_observation()
        return obs_dict

    @property
    def observation_space(self) -> spaces.Space:
        """Return the combined Gymnasium observation space."""
        return spaces.Dict(
            {
                ev_id: spaces.Dict({obs_id: om.obs_space for obs_id, om in om_dict.items()})
                for ev_id, om_dict in self._obs_managers.items()
            }
        )

    def reset(self, ego_vehicles: Mapping[str, Any]) -> None:
        """Re-initialize managers and attach new ego vehicle actors."""
        self._init_obs_managers()

        for ev_id, ev_actor in ego_vehicles.items():
            for om in self._obs_managers[ev_id].values():
                om.attach_ego_vehicle(ev_actor)

    def clean(self) -> None:
        """Clean up all managed `ObsManager`s and their resources."""
        for om_dict in self._obs_managers.values():
            for om in om_dict.values():
                om.clean()
        self._obs_managers = {}

    def _init_obs_managers(self) -> None:
        """Instantiate all observation managers according to the configuration."""
        self._obs_managers = {
            ev_id: {
                obs_id: self._resolve_manager_class(obs_cfg["module"])(obs_cfg) for obs_id, obs_cfg in obs_cfgs.items()
            }
            for ev_id, obs_cfgs in self._obs_configs.items()
        }

    @staticmethod
    def _resolve_manager_class(path: str) -> type[BaseObservation]:  # ★ 追加
        """Return BaseObservation class from 'module:Class' (class optional)."""
        base_pkg = "carla_gym.engine.observation."
        module_path, _, class_name = path.partition(":")
        class_name = class_name or "ObsManager"

        module = import_module(f"{base_pkg}{module_path}")
        try:
            return getattr(module, class_name)
        except AttributeError as exc:  # 明示的にエラーを変換
            raise ImportError(f"Class '{class_name}' not found in module '{module.__name__}'") from exc
