"""Leaderboard 2.0 Scenario Loader for CarlaGym.

This module provides utilities to automatically discover and load
scenarios from leaderboard_2_0 XML structure, enabling seamless
integration with the existing CarlaGym TaskConfig system.

Features:
--------
1. Auto-discovery of XML routes from leaderboard_2_0 directory structure
2. Scenario type filtering (Accident, BlockedIntersection, etc.)
3. Random route selection within scenario types
4. Integration with existing TaskConfig workflow
"""

from __future__ import annotations

import os
import glob
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import carla
from carla_gym.envs.task_config import TaskConfig, ScenarioConfig
from carla_gym.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LeaderboardRoute:
    """Represents a single route from Leaderboard 2.0 XML files."""

    route_id: str
    scenario_type: str
    town: str
    xml_file: str
    waypoints: List[Tuple[float, float, float]]
    scenarios: List[ScenarioConfig]
    weather: Optional[Dict] = None


class LeaderboardScenarioLoader:
    """Loader for Leaderboard 2.0 scenario XML files with auto-discovery."""

    # Available scenario types in leaderboard_2_0
    SCENARIO_TYPES = [
        "Accident",
        "AccidentTwoWays",
        "BlockedIntersection",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "ControlLoss",
        "DynamicObjectCrossing",
        "EnterActorFlow",
        "HardBreakRoute",
        "HazardAtSideLane",
        "InvadingTurn",
        "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority",
        "OtherLeadingVehicle",
        "ParkingCutIn",
        "ParkingExit",
        "PedestrianCrossing",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleOpensDoor",
        "YieldToEmergencyVehicle",
    ]

    def __init__(self, leaderboard_data_path: str):
        """Initialize the loader with leaderboard_2_0 data path.

        Args:
            leaderboard_data_path: Path to leaderboard_2_0/data directory
        """
        self.data_path = Path(leaderboard_data_path)
        self._route_cache: Dict[str, List[LeaderboardRoute]] = {}
        self._scan_routes()

    def _scan_routes(self) -> None:
        """Scan all XML files in the leaderboard data directory."""
        logger.info(f"Scanning routes in {self.data_path}")

        # Scan town directories (e.g., 50x36_town_13, 50x38_town_12)
        town_dirs = [d for d in self.data_path.iterdir() if d.is_dir() and "town" in d.name.lower()]

        for town_dir in town_dirs:
            town_name = self._extract_town_name(town_dir.name)

            # Scan scenario type directories
            for scenario_dir in town_dir.iterdir():
                if not scenario_dir.is_dir():
                    continue

                scenario_type = scenario_dir.name
                if scenario_type not in self.SCENARIO_TYPES:
                    continue

                # Find all XML files in this scenario directory
                xml_files = list(scenario_dir.glob("*.xml"))
                routes = []

                for xml_file in xml_files:
                    try:
                        route = self._parse_xml_route(xml_file, town_name, scenario_type)
                        if route:
                            routes.append(route)
                    except Exception as e:
                        logger.warning(f"Failed to parse {xml_file}: {e}")

                if routes:
                    cache_key = f"{town_name}_{scenario_type}"
                    self._route_cache[cache_key] = routes
                    logger.info(f"Loaded {len(routes)} routes for {cache_key}")

    def _extract_town_name(self, dir_name: str) -> str:
        """Extract CARLA town name from directory name.

        Args:
            dir_name: Directory name like '50x36_town_13'

        Returns:
            Town name like 'Town13'
        """
        # Extract town number from directory name
        if "town" in dir_name.lower():
            parts = dir_name.lower().split("_")
            for part in parts:
                if part.startswith("town"):
                    town_num = part.replace("town", "").strip()
                    return f"Town{town_num.zfill(2)}"
        return "Town01"  # fallback

    def _parse_xml_route(self, xml_file: Path, town: str, scenario_type: str) -> Optional[LeaderboardRoute]:
        """Parse a single XML route file.

        Args:
            xml_file: Path to XML file
            town: CARLA town name
            scenario_type: Type of scenario

        Returns:
            Parsed LeaderboardRoute or None if parsing fails
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Handle both single route and routes container
            route_elem = root if root.tag == "route" else root.find("route")
            if route_elem is None:
                return None

            route_id = route_elem.attrib.get("id", xml_file.stem)

            # Parse waypoints
            waypoints = []
            waypoints_elem = route_elem.find("waypoints")
            if waypoints_elem is not None:
                for pos_elem in waypoints_elem.findall("position"):
                    x = float(pos_elem.attrib["x"])
                    y = float(pos_elem.attrib["y"])
                    z = float(pos_elem.attrib.get("z", 0.0))
                    waypoints.append((x, y, z))

            # Parse scenarios
            scenarios = []
            scenarios_elem = route_elem.find("scenarios")
            if scenarios_elem is not None:
                for scenario_elem in scenarios_elem.findall("scenario"):
                    try:
                        scenario_config = ScenarioConfig.from_xml(scenario_elem)
                        scenarios.append(scenario_config)
                    except Exception as e:
                        logger.warning(f"Failed to parse scenario in {xml_file}: {e}")

            # Parse weather if available
            weather = None
            weathers_elem = route_elem.find("weathers")
            if weathers_elem is not None:
                weather_elem = weathers_elem.find("weather")
                if weather_elem is not None:
                    weather = dict(weather_elem.attrib)

            return LeaderboardRoute(
                route_id=route_id,
                scenario_type=scenario_type,
                town=town,
                xml_file=str(xml_file),
                waypoints=waypoints,
                scenarios=scenarios,
                weather=weather,
            )

        except ET.ParseError as e:
            logger.error(f"XML parse error in {xml_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing {xml_file}: {e}")
            return None

    def get_available_scenario_types(self, town: Optional[str] = None) -> List[str]:
        """Get list of available scenario types.

        Args:
            town: Optional town filter

        Returns:
            List of available scenario types
        """
        if town:
            return [key.split("_", 1)[1] for key in self._route_cache.keys() if key.startswith(town)]
        else:
            return list(set(key.split("_", 1)[1] for key in self._route_cache.keys()))

    def get_available_towns(self) -> List[str]:
        """Get list of available towns.

        Returns:
            List of available CARLA town names
        """
        return list(set(key.split("_", 1)[0] for key in self._route_cache.keys()))

    def get_routes_by_type(self, scenario_type: str, town: Optional[str] = None) -> List[LeaderboardRoute]:
        """Get all routes of a specific scenario type.

        Args:
            scenario_type: Type of scenario to filter
            town: Optional town filter

        Returns:
            List of matching routes
        """
        routes = []
        for cache_key, cached_routes in self._route_cache.items():
            key_town, key_type = cache_key.split("_", 1)
            if key_type == scenario_type and (town is None or key_town == town):
                routes.extend(cached_routes)
        return routes

    def get_random_route(
        self, scenario_type: Optional[str] = None, town: Optional[str] = None
    ) -> Optional[LeaderboardRoute]:
        """Get a random route matching the criteria.

        Args:
            scenario_type: Optional scenario type filter
            town: Optional town filter

        Returns:
            Random matching route or None
        """
        if scenario_type:
            routes = self.get_routes_by_type(scenario_type, town)
        else:
            routes = []
            for cached_routes in self._route_cache.values():
                if town:
                    routes.extend([r for r in cached_routes if r.town == town])
                else:
                    routes.extend(cached_routes)

        return random.choice(routes) if routes else None

    def create_task_config(
        self, route: LeaderboardRoute, num_npc_vehicles: int = 0, num_npc_walkers: int = 0, weather: str = "ClearNoon"
    ) -> TaskConfig:
        """Create a TaskConfig from a LeaderboardRoute.

        Args:
            route: LeaderboardRoute to convert
            num_npc_vehicles: Number of NPC vehicles
            num_npc_walkers: Number of NPC walkers
            weather: Weather condition

        Returns:
            TaskConfig instance
        """
        return TaskConfig(
            weather=weather,
            map_name=route.town,
            num_npc_vehicles=num_npc_vehicles,
            num_npc_walkers=num_npc_walkers,
            route_file=route.xml_file,
            route_id=route.route_id,
            scenarios=route.scenarios,
            keypoints=route.waypoints,
        )

    def create_random_task_config(
        self, scenario_type: Optional[str] = None, town: Optional[str] = None, **task_kwargs
    ) -> Optional[TaskConfig]:
        """Create a TaskConfig from a random route.

        Args:
            scenario_type: Optional scenario type filter
            town: Optional town filter
            **task_kwargs: Additional TaskConfig parameters

        Returns:
            Random TaskConfig or None
        """
        route = self.get_random_route(scenario_type, town)
        if not route:
            return None

        return self.create_task_config(route, **task_kwargs)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded routes.

        Returns:
            Dictionary with route statistics
        """
        stats = {
            "total_routes": sum(len(routes) for routes in self._route_cache.values()),
            "scenario_types": len(self.get_available_scenario_types()),
            "towns": len(self.get_available_towns()),
        }

        # Per-scenario-type counts
        for scenario_type in self.get_available_scenario_types():
            routes = self.get_routes_by_type(scenario_type)
            stats[f"{scenario_type}_count"] = len(routes)

        return stats


def create_leaderboard_loader(leaderboard_2_0_path: Optional[str] = None) -> LeaderboardScenarioLoader:
    """Factory function to create a LeaderboardScenarioLoader.

    Args:
        leaderboard_2_0_path: Path to leaderboard_2_0 directory.
                             If None, tries to auto-detect.

    Returns:
        LeaderboardScenarioLoader instance

    Raises:
        FileNotFoundError: If leaderboard_2_0 data directory not found
    """
    if leaderboard_2_0_path is None:
        # Try to auto-detect leaderboard_2_0 path
        possible_paths = [
            "carla_benchmarks/leaderboard_2_0/data",
            "../carla_benchmarks/leaderboard_2_0/data",
            "../../carla_benchmarks/leaderboard_2_0/data",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                leaderboard_2_0_path = path
                break

        if leaderboard_2_0_path is None:
            raise FileNotFoundError(
                "Could not find leaderboard_2_0 data directory. Please specify the path explicitly."
            )

    data_path = Path(leaderboard_2_0_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Leaderboard data path does not exist: {data_path}")

    return LeaderboardScenarioLoader(str(data_path))
