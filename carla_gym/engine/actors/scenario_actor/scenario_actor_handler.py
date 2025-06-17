"""Light-weight bridge between CarlaEnv and CARLA ScenarioRunner.

Responsibilities
----------------
1. Load traffic scenarios (Route XML, Leaderboard format).
2. Tick the scenario tree every CarlaEnv step.
3. Dynamically build new atomic scenarios & spawn parked vehicles
   at ~1 Hz without extra threads.
"""

from __future__ import annotations

import os
import time
from typing import Final

import carla
import py_trees
from leaderboard.scenarios.route_scenario import RouteScenario as _LBRouteScenario
from leaderboard.utils.route_parser import RouteParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.envs.task_config import TaskConfig


class _RouteScenarioReuseEgo(_LBRouteScenario):
    """RouteScenario variant that re-uses the already spawned *hero* vehicle."""

    def _spawn_ego_vehicle(self):
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            raise RuntimeError("Hero vehicle (role_name='hero') not found.")
        return hero


class ScenarioActorHandler:
    """Manage Leaderboard traffic scenarios inside CarlaEnv."""

    _WATCHDOG_TIMEOUT: Final[float] = 20.0
    _BUILD_INTERVAL_SEC: Final[float] = 1.0  # frequency for build/spawn

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, client: carla.Client):
        self._client = client
        self._world = client.get_world()

        # Register helpers
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(self._world)

        # ScenarioManager in *non-blocking* mode
        self._manager = ScenarioManager(
            debug_mode=False,
            sync_mode=False,  # CarlaEnv calls world.tick()
            timeout=self._WATCHDOG_TIMEOUT,
        )

        # Internal state
        self._scenario = None
        self._last_sim_time = float("-inf")
        self._last_build_time = float("-inf")

        os.environ["SCENARIO_RUNNER_ROOT"] = (
            f"{CARLA_GYM_ROOT_DIR.as_posix()}/packages/carla_garage/scenario_runner_autopilot"
        )

    # ------------------------------------------------------------------ #
    # Public API – called by CarlaEnv                                    #
    # ------------------------------------------------------------------ #
    def reset(self, task_cfg: TaskConfig) -> None:
        """Load (or skip) traffic scenario described in *task_cfg*."""
        self.clean()

        if not task_cfg.route_file:
            return  # no scenario requested

        # --- ensure hero exists & register to DataProvider ----------------
        hero = None
        for actor in self._world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "hero":
                hero = actor
                break
        if hero is None:
            # defer scenario loading until next episode
            print("ScenarioHandler: hero vehicle not found; skip scenario loading.")
            return

        CarlaDataProvider.register_actor(hero, hero.get_transform())
        CarlaDataProvider._carla_actor_pool[hero.id] = hero

        # Parse routes (first match if route_id is None)
        route_confs = RouteParser.parse_routes_file(task_cfg.route_file, task_cfg.route_id or "")
        if not route_confs:
            raise ValueError(f"Route id '{task_cfg.route_id}' not found in {task_cfg.route_file}")

        # Instantiate RouteScenario that reuses existing ego
        self._scenario = _RouteScenarioReuseEgo(self._world, route_confs[0], debug_mode=0)
        self._manager.load_scenario(self._scenario)

        # Manual execution mode
        self._manager._watchdog = Watchdog(self._WATCHDOG_TIMEOUT)
        self._manager._watchdog.start()
        self._manager._running = True
        self._manager.start_system_time = time.time()
        self._manager.start_game_time = GameTime.get_time()

        self._last_sim_time = float("-inf")
        self._last_build_time = float("-inf")

    def tick(self, timestamp: carla.Timestamp) -> None:
        """Advance scenario state; call every CarlaEnv step."""
        if not getattr(self._manager, "_running", False):
            return

        sim_t = timestamp.elapsed_seconds
        if sim_t <= self._last_sim_time:  # duplicate guard
            return
        self._last_sim_time = sim_t

        # --- dynamic build / spawn (@~1 Hz) ----------------------------
        if self._scenario and (sim_t - self._last_build_time) >= self._BUILD_INTERVAL_SEC:
            hero = CarlaDataProvider.get_hero_actor()
            if hero:
                # Build new atomic scenarios close to ego
                self._scenario.build_scenarios(hero, debug=False)
                # Spawn parked vehicles when close enough
                self._scenario.spawn_parked_vehicles(hero)
            self._last_build_time = sim_t

        # Watchdog / timers / providers
        self._manager._watchdog.update()
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()

        # Tick scenario tree once
        if self._manager.scenario_tree:
            self._manager.scenario_tree.tick_once()
            if self._manager.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._manager.stop_scenario()

    def clean(self) -> None:
        """Safely terminate current scenario and destroy actors."""
        if self._scenario:
            self._manager.cleanup()
            self._scenario = None
