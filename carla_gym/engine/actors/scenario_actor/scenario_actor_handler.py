"""Enhanced bridge between CarlaEnv and CARLA ScenarioRunner with Leaderboard 2.0 support.

Responsibilities
----------------
1. Load traffic scenarios (Route XML, Leaderboard 2.0 format).
2. Tick the scenario tree every CarlaEnv step.
3. Dynamically build new atomic scenarios & spawn parked vehicles
   at ~1 Hz without extra threads.
4. Integrate stuck detection from scenario_manager_local.py
"""

from __future__ import annotations

import os
import time
from typing import Final

import carla
import py_trees
from carla_gym.external.srunner.scenarios.route_scenario import RouteScenario as _LBRouteScenario
from carla_gym.external.srunner.tools.route_parser import RouteParser
from carla_gym.external.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from carla_gym.external.srunner.scenariomanager.scenario_manager import ScenarioManager
from carla_gym.external.srunner.scenariomanager.timer import GameTime
from carla_gym.external.srunner.scenariomanager.watchdog import Watchdog

from carla_gym.envs.task_config import TaskConfig


class _RouteScenarioReuseEgo(_LBRouteScenario):
    """RouteScenario variant that re-uses the already spawned *hero* vehicle."""

    def _spawn_ego_vehicle(self):
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            raise RuntimeError("Hero vehicle (role_name='hero') not found.")
        return hero


class ScenarioActorHandler:
    """Manage Leaderboard 2.0 traffic scenarios inside CarlaEnv with stuck detection."""

    _WATCHDOG_TIMEOUT: Final[float] = 20.0
    _BUILD_INTERVAL_SEC: Final[float] = 1.0  # frequency for build/spawn

    # Stuck detection constants (from scenario_manager_local.py)
    STUCK_DISTANCE_THRESHOLD_METERS: Final[float] = 0.5
    STUCK_DURATION_THRESHOLD_SECONDS: Final[float] = 10.0
    MAX_CONSECUTIVE_STUCK_TICKS: Final[int] = 50

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

        # Stuck detection variables (integrated from scenario_manager_local.py)
        self._scenario_aborted_due_to_stuck = False
        self._agent_last_significant_move_pos = None
        self._agent_last_significant_move_time = None
        self._is_stuck = False
        self._stuck_check_initialized = False
        self._consecutive_stuck_ticks = 0

        os.environ["SCENARIO_RUNNER_ROOT"] = os.path.dirname(os.path.abspath(__file__ + "/../../../external/srunner"))

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

        # Reset stuck detection variables for the new scenario
        self._scenario_aborted_due_to_stuck = False
        self._is_stuck = False
        self._consecutive_stuck_ticks = 0
        self._stuck_check_initialized = False
        self._agent_last_significant_move_pos = None
        self._agent_last_significant_move_time = None

    def tick(self, timestamp: carla.Timestamp, ego_action: carla.VehicleControl = None) -> None:
        """Advance scenario state with stuck detection; call every CarlaEnv step."""
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

        # Stuck detection (integrated from scenario_manager_local.py)
        if self._scenario and ego_action is not None:
            self._check_and_handle_stuck_status(timestamp, ego_action)

        # Tick scenario tree once
        if self._manager.scenario_tree:
            self._manager.scenario_tree.tick_once()
            if self._manager.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._manager.stop_scenario()

        # Stop scenario if stuck is detected
        if self._scenario_aborted_due_to_stuck:
            self._manager.stop_scenario()

    def clean(self) -> None:
        """Safely terminate current scenario and destroy actors."""
        if self._scenario:
            self._manager.cleanup()
            self._scenario = None

    # ------------------------------------------------------------------ #
    # Stuck Detection Methods (from scenario_manager_local.py)           #
    # ------------------------------------------------------------------ #

    def _initialize_stuck_detection(self, current_pos: carla.Location, current_time: float) -> None:
        """Initialize the variables for stuck detection on the first relevant tick."""
        self._agent_last_significant_move_pos = current_pos
        self._agent_last_significant_move_time = current_time
        self._stuck_check_initialized = True

    def _has_vehicle_moved_sufficiently(self, current_pos: carla.Location) -> bool:
        """Check if the vehicle has moved a significant distance since the last check."""
        if self._agent_last_significant_move_pos is None:
            return False
        distance_moved = current_pos.distance(self._agent_last_significant_move_pos)
        return distance_moved > self.STUCK_DISTANCE_THRESHOLD_METERS

    def _update_status_for_moving_vehicle(
        self, current_pos: carla.Location, current_time: float, timestamp: carla.Timestamp
    ) -> None:
        """Update agent status when it has moved sufficiently (normal operation path)."""
        self._agent_last_significant_move_pos = current_pos
        self._agent_last_significant_move_time = current_time
        if self._is_stuck:
            print(f"[ScenarioActorHandler] Vehicle unstuck at tick {timestamp.frame}.")
        self._is_stuck = False
        self._consecutive_stuck_ticks = 0

    def _check_and_handle_stuck_status(self, timestamp: carla.Timestamp, ego_action: carla.VehicleControl) -> None:
        """Check if the ego vehicle is stuck and update scenario status accordingly."""
        if not self._stuck_check_initialized:
            hero = CarlaDataProvider.get_hero_actor()
            if hero:
                current_pos_init = hero.get_location()
                current_time_init = timestamp.elapsed_seconds
                self._initialize_stuck_detection(current_pos_init, current_time_init)
            return

        hero = CarlaDataProvider.get_hero_actor()
        if not hero:
            return

        current_pos = hero.get_location()
        current_time = timestamp.elapsed_seconds

        if self._has_vehicle_moved_sufficiently(current_pos):
            self._update_status_for_moving_vehicle(current_pos, current_time, timestamp)
        else:
            self._handle_vehicle_potentially_stuck(current_time, ego_action, timestamp)

    def _handle_vehicle_potentially_stuck(
        self, current_time: float, ego_action: carla.VehicleControl, timestamp: carla.Timestamp
    ) -> None:
        """Handle the logic when the vehicle has not moved significantly."""
        if ego_action is None:
            self._reset_stuck_status_due_to_no_action(current_time, timestamp)
            return

        is_trying_to_move = ego_action.throttle > 0.1 and ego_action.brake < 0.5
        if is_trying_to_move:
            self._check_stuck_while_trying_to_move(current_time, ego_action, timestamp)
        else:
            self._reset_stuck_status_when_not_moving(current_time, timestamp)

    def _reset_stuck_status_due_to_no_action(self, current_time: float, timestamp: carla.Timestamp) -> None:
        """Reset the stuck status if the agent's action is unavailable."""
        if self._is_stuck:
            print(f"[ScenarioActorHandler] Agent action unavailable, resetting stuck state at tick {timestamp.frame}.")
        self._is_stuck = False
        self._consecutive_stuck_ticks = 0
        self._agent_last_significant_move_time = current_time

    def _check_stuck_while_trying_to_move(
        self, current_time: float, ego_action: carla.VehicleControl, timestamp: carla.Timestamp
    ) -> None:
        """Check for stuck condition when the agent is actively trying to move."""
        time_since_last_move = current_time - self._agent_last_significant_move_time

        if time_since_last_move > self.STUCK_DURATION_THRESHOLD_SECONDS:
            if not self._is_stuck:
                print(
                    f"[ScenarioActorHandler] WARNING: Vehicle might be stuck! "
                    f"(throttle: {ego_action.throttle:.2f}, brake: {ego_action.brake:.2f}) "
                    f"no significant movement for {time_since_last_move:.2f} seconds "
                    f"at tick {timestamp.frame}."
                )
                self._is_stuck = True

            self._consecutive_stuck_ticks += 1
            if self._consecutive_stuck_ticks > self.MAX_CONSECUTIVE_STUCK_TICKS:
                print(
                    f"[ScenarioActorHandler] ERROR: Vehicle confirmed stuck for {self._consecutive_stuck_ticks} ticks "
                    f"while trying to move. Aborting scenario at tick {timestamp.frame}."
                )
                self._scenario_aborted_due_to_stuck = True

    def _reset_stuck_status_when_not_moving(self, current_time: float, timestamp: carla.Timestamp) -> None:
        """Reset the stuck status when the agent is not actively trying to move."""
        if self._is_stuck:
            print(f"[ScenarioActorHandler] Vehicle no longer stuck (not accelerating) at tick {timestamp.frame}.")
        self._is_stuck = False
        self._consecutive_stuck_ticks = 0
        self._agent_last_significant_move_time = current_time

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def is_scenario_aborted_due_to_stuck(self) -> bool:
        """Return whether the scenario was aborted due to stuck detection."""
        return self._scenario_aborted_due_to_stuck
