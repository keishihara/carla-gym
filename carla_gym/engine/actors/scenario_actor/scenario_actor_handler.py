from __future__ import annotations

import time

import carla
import py_trees

# srunner components for scenario execution
# Note: Ensure that the srunner module from carla_garage is in the PYTHONPATH
try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.scenario_manager import ScenarioManager
    from srunner.scenariomanager.timer import GameTime
    from srunner.scenariomanager.watchdog import Watchdog
    from srunner.scenarios.route_scenario import RouteScenario
    from srunner.tools.route_parser import RouteParser
except ImportError as e:
    raise ImportError(
        "Failed to import srunner. Please make sure srunner is installed, e.g., "
        "by running 'pip install -e .' in 'packages/carla_garage/scenario_runner'."
    ) from e


class ScenarioActorHandler:
    """
    This handler manages the lifecycle of scenarios from carla-garage (leaderboard).
    It initializes, resets, ticks, and cleans scenarios based on the provided route
    and scenario definitions from a task configuration file.
    """

    def __init__(self, client: carla.Client):
        self._client = client
        self._world = client.get_world()
        self._scenario = None

        # Setup srunner's helper modules
        CarlaDataProvider.set_client(self._client)
        CarlaDataProvider.set_world(self._world)

        # The scenario manager should not handle world ticks, as carla-gym does.
        # The timeout should be a float.
        self._scenario_manager = ScenarioManager(debug_mode=False, sync_mode=False, timeout=20.0)

    def reset(self, task_config: dict) -> None:
        """
        Parses route and scenario configurations and loads the scenario.
        This method expects 'route_file' and 'route_id' in the task_config.
        """
        self.clean()

        route_file = task_config.get("route_file")
        if not route_file:
            # If no route file is provided in the task, do nothing.
            return

        route_id = task_config.get("route_id", "")  # Empty means all routes, we'll take the first

        # Parse the route and scenario configurations from the XML file.
        # We expect one configuration for a specific route_id, so we take the first one.
        route_configs = RouteParser.parse_routes_file(route_file, route_id)
        if not route_configs:
            raise ValueError(f"Could not find route with id '{route_id}' in file '{route_file}'")
        config = route_configs[0]

        # Prepare and load the RouteScenario, which will manage all atomic scenarios.
        self._scenario = RouteScenario(world=self._world, config=config, debug_mode=False)
        self._scenario_manager.load_scenario(self._scenario)

        # Manually start the scenario execution tracking, bypassing the blocking run_scenario()
        if self._scenario_manager.scenario is not None:
            self._scenario_manager._watchdog = Watchdog(float(self._scenario_manager._timeout))
            self._scenario_manager._watchdog.start()
            self._scenario_manager._running = True
            self._scenario_manager.start_system_time = time.time()
            self._scenario_manager.start_game_time = GameTime.get_time()

    def tick(self, timestamp: carla.Timestamp) -> None:
        """
        Ticks the scenario to progress its state, without controlling the ego vehicle.
        """
        if not (
            self._scenario_manager and hasattr(self._scenario_manager, "_running") and self._scenario_manager._running
        ):
            return

        manager = self._scenario_manager

        # 1. Update watchdog and timers
        manager._watchdog.update()
        if manager._timestamp_last_run >= timestamp.elapsed_seconds:
            return
        manager._timestamp_last_run = timestamp.elapsed_seconds

        # 2. Update data providers
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()

        # 3. Tick the scenario tree
        # The agent part is intentionally skipped as carla-gym handles the ego vehicle.
        if manager.scenario_tree:
            manager.scenario_tree.tick_once()

        # 4. Check scenario status
        if manager.scenario_tree and manager.scenario_tree.status != py_trees.common.Status.RUNNING:
            manager._running = False

        # The world tick part from the original _tick_scenario is also skipped,
        # as carla-gym handles it in the main step loop.

    def clean(self) -> None:
        """
        Cleans up the scenario manager and all associated actors.
        """
        if self._scenario_manager:
            # This will terminate the scenario and destroy all its actors.
            self._scenario_manager.cleanup()
        self._scenario = None
