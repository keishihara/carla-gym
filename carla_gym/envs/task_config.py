"""Task and Scenario configuration utilities for carla_gym.

This module defines dataclass-based representations for:
    • EgoVehiclesConfig – ego vehicle actor settings.
    • ScenarioConfig    – traffic event specification (CARLA ScenarioRunner).
    • TaskConfig        – one environment episode configuration.
    • TaskSet           – a collection helper for multiple TaskConfig.

The goal is to improve readability and type safety compared with plain
nested dictionaries, while providing ``to_dict`` adapters so the existing
``CarlaEnv`` implementation can continue to operate unchanged.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import carla
import numpy as np
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.tools.route_parser import RouteParser

from carla_gym.utils.logger import setup_logger

logger = setup_logger(__name__)


__all__ = [
    "EgoVehiclesConfig",
    "ScenarioConfig",
    "TaskConfig",
    "TaskSet",
    "MAP_NAMES_SHORT",
    "MAP_NAMES_FULL",
    "MAX_NUM_NPC_VEHICLES",
    "MAX_NUM_NPC_WALKERS",
]


MAP_NAMES_SHORT = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
]
MAP_NAMES_FULL = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town12",
    "Town13",
    "Town15",
]

MAX_NUM_NPC_VEHICLES = {
    "Town01": 120,
    "Town02": 70,
    "Town03": 70,
    "Town04": 150,
    "Town05": 120,
    "Town06": 120,
    "Town07": 100,
    "Town10H": 100,
    "Town12": 10,
    "Town13": 10,
    "Town15": 70,
}
MAX_NUM_NPC_WALKERS = {
    "Town01": 120,
    "Town02": 70,
    "Town03": 70,
    "Town04": 80,
    "Town05": 120,
    "Town06": 80,
    "Town07": 100,
    "Town10H": 100,
    "Town12": 10,
    "Town13": 10,
    "Town15": 70,
}

WEATHERS = [
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "SoftRainNoon",
    "MidRainyNoon",
    "HardRainNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "SoftRainSunset",
    "MidRainSunset",
    "HardRainSunset",
]


def get_weathers(weather_group: str) -> list[str]:
    if "dynamic" in weather_group:
        # For dynamic weather, create single task
        weathers = [weather_group]
    elif weather_group == "new":
        weathers = ["SoftRainSunset", "WetSunset"]
    elif weather_group == "train":
        weathers = ["ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset"]
    elif weather_group == "all":
        weathers = WEATHERS
    else:
        raise ValueError(f"Invalid weather group: {weather_group}")
    return weathers


# ---------------------------------------------------------------------------
#  Basic building blocks
# ---------------------------------------------------------------------------


@dataclass
class EgoVehiclesConfig:
    """Configuration wrapper for ego vehicle(s).

    Currently we only support a single *hero* vehicle with ChauffeurNet-style
    birdview observations, but the structure leaves room for extension.
    """

    actors: dict[str, dict] = field(default_factory=lambda: {"hero": {"model": "vehicle.lincoln.mkz_2017"}})
    routes: dict[str, Sequence] = field(default_factory=lambda: {"hero": []})
    endless: dict[str, bool] = field(default_factory=lambda: {"hero": True})

    # ---------------------------------------------------------------------
    #  Conversion helpers
    # ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a pure-dict representation compatible with *CarlaEnv*."""

        return {
            "actors": self.actors,
            "routes": self.routes,
            "endless": self.endless,
        }


@dataclass(kw_only=True)
class ScenarioConfig:
    """Dataclass representation of an individual ScenarioRunner scenario."""

    name: str
    type: str
    trigger_point: carla.Transform
    parameters: dict[str, Any] = field(default_factory=dict)
    other_actors: list[Any] = field(default_factory=list)

    # ---------------------------------------------------------------------
    #  Factory helpers
    # ---------------------------------------------------------------------

    @classmethod
    def from_srunner(cls, sc_conf: ScenarioConfiguration) -> ScenarioConfig:
        """Convert srunner.scenarioconfigs.ScenarioConfiguration → ScenarioConfig."""
        trigger = sc_conf.trigger_points[0] if sc_conf.trigger_points else carla.Transform()
        return cls(
            name=sc_conf.name or "",
            type=sc_conf.type or "",
            trigger_point=trigger,
            parameters=sc_conf.other_parameters.copy(),
            other_actors=sc_conf.other_actors or [],
        )

    @classmethod
    def from_xml(cls, xml_elem: ET.Element) -> ScenarioConfig:
        """Parse a <scenario> node from a CARLA Leaderboard routes XML file."""

        name = xml_elem.attrib.get("name", "")
        type_ = xml_elem.attrib.get("type", "")

        trigger_elem = xml_elem.find("trigger_point")
        if trigger_elem is None:
            raise ValueError(f"Scenario '{name}' is missing <trigger_point> tag")

        tp_attrib = trigger_elem.attrib
        loc = carla.Location(
            x=float(tp_attrib["x"]),
            y=float(tp_attrib["y"]),
            z=float(tp_attrib.get("z", 0.0)),
        )
        rot = carla.Rotation(yaw=float(tp_attrib.get("yaw", 0.0)))
        transform = carla.Transform(loc, rot)

        # Parse additional parameter tags (exclude trigger_point)
        params: dict[str, Any] = {}
        for child in xml_elem:
            if child is trigger_elem:
                continue
            # Use tag name as key; coerce attribute/element text to float if possible
            if child.attrib:
                # e.g. <direction value="left"/>
                # When 'value' attribute exists, use it, else store whole attrib dict
                if "value" in child.attrib and len(child.attrib) == 1:
                    params[child.tag] = _auto_cast(child.attrib["value"])
                else:
                    params[child.tag] = {k: _auto_cast(v) for k, v in child.attrib.items()}
            elif child.text and child.text.strip():
                params[child.tag] = _auto_cast(child.text.strip())

        return cls(name=name, type=type_, trigger_point=transform, parameters=params)

    # ------------------------------------------------------------------
    #  Conversion helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dict (carla types converted to python primitives)."""

        t = self.trigger_point
        return {
            "name": self.name,
            "type": self.type,
            "trigger_point": {
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"yaw": t.rotation.yaw},
            },
            "parameters": self.parameters,
            "other_actors": [str(a) for a in self.other_actors],
        }


# ---------------------------------------------------------------------------
#  TaskConfig – represents a single episode setting
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """One environment task (episode) configuration."""

    weather: str
    map_name: str
    num_npc_vehicles: int | tuple[int, int]
    num_npc_walkers: int | tuple[int, int]
    ego_vehicles: EgoVehiclesConfig = field(default_factory=EgoVehiclesConfig)
    scenarios: list[ScenarioConfig] = field(default_factory=list)
    # Route waypoints kept as plain floats for pickling safety. Each entry is
    # (x, y, z).  They are converted to ``carla.Transform`` objects by
    # :py:meth:`resolve_routes` immediately before the episode starts.
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)
    seed: int | None = None

    route_file: str | None = None
    route_id: str | None = None

    # ------------------------------------------------------------------
    #  Post-processing
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Normalise NPC ranges to tuple[int, int]
        self.num_npc_vehicles = _normalise_range(self.num_npc_vehicles)
        self.num_npc_walkers = _normalise_range(self.num_npc_walkers)

        self._load_route_file_if_available()

    def _load_route_file_if_available(self) -> None:
        if not self.route_file:
            return

        route_confs = RouteParser.parse_routes_file(self.route_file)

        if self.route_id is None:
            logger.warning("route_id is not set, using the first route id")
            self.route_id = route_confs[0].name.replace("RouteScenario_", "")

        route_conf = next((rc for rc in route_confs if rc.name == f"RouteScenario_{self.route_id}"), None)
        if route_conf is None:
            raise ValueError(f"Route {self.route_id} not found in {self.route_file}")

        self.scenarios = [ScenarioConfig.from_srunner(sc) for sc in route_conf.scenario_configs]

        # Store raw floats only (conversion deferred)
        self.keypoints = [(p.x, p.y, p.z) for p in route_conf.keypoints]
        # Routes will be filled later by resolve_routes()
        self.ego_vehicles = EgoVehiclesConfig()

    @classmethod
    def from_route_file(cls, route_file: str, route_id: str) -> TaskConfig:
        """Create TaskConfig from route file and route id."""
        route_confs = RouteParser.parse_routes_file(route_file)
        route_conf = next((rc for rc in route_confs if rc.name == f"RouteScenario_{route_id}"), None)
        if route_conf is None:
            raise ValueError(f"Route {route_id} not found in {route_file}")

        scenarios = [ScenarioConfig.from_xml(sc) for sc in route_conf.scenario_configs]

        # Defer Transform creation – keep floats only
        float_pts = [(p.x, p.y, p.z) for p in route_conf.keypoints]
        ego_conf = EgoVehiclesConfig()

        return cls(
            weather=route_conf.weather,
            map_name=route_conf.town,
            num_npc_vehicles=0,
            num_npc_walkers=0,
            ego_vehicles=ego_conf,
            route_file=route_file,
            route_id=route_id,
            scenarios=scenarios,
            keypoints=float_pts,
        )

    # ------------------------------------------------------------------
    #  Behaviour helpers
    # ------------------------------------------------------------------

    def sample_npcs(self, rng: np.random.Generator | None = None) -> tuple[int, int]:
        """Sample concrete numbers of NPC vehicles & walkers within the range."""

        rng = rng or np.random.default_rng()
        veh = _sample_range(self.num_npc_vehicles, rng)
        wlk = _sample_range(self.num_npc_walkers, rng)
        return veh, wlk

    @staticmethod
    def sample(
        *,
        n: int = 1,
        map_names: str | Sequence[str] = MAP_NAMES_SHORT,
        weathers: str | Sequence[str] = ("dynamic_1.0",),
        num_npc_vehicles: int | tuple[int, int] | None = None,
        num_npc_walkers: int | tuple[int, int] | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> TaskConfig | list[TaskConfig]:
        """Return one TaskConfig with fully randomised parameters."""
        rng = rng or np.random.default_rng(seed=seed)

        def _sample_one(idx: int = 0) -> TaskConfig:
            weather = weathers if isinstance(weathers, str) else rng.choice(list(weathers))
            town = map_names if isinstance(map_names, str) else rng.choice(list(map_names))
            npc_veh = _sample_range(num_npc_vehicles or (0, MAX_NUM_NPC_VEHICLES[town]), rng)
            npc_wlk = _sample_range(num_npc_walkers or (0, MAX_NUM_NPC_WALKERS[town]), rng)

            return TaskConfig(
                weather=weather,
                map_name=town,
                num_npc_vehicles=npc_veh,
                num_npc_walkers=npc_wlk,
                seed=seed + idx if isinstance(seed, int) else None,
            )

        if n == 1:
            return _sample_one()
        return [_sample_one(i) for i in range(n)]

    # ------------------------------------------------------------------
    #  Conversion helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a pickle-safe plain dict (no CARLA objects)."""

        return {
            "weather": self.weather,
            "map_name": self.map_name,
            "num_npc_vehicles": self.num_npc_vehicles,
            "num_npc_walkers": self.num_npc_walkers,
            "ego_vehicles": self.ego_vehicles.to_dict(),
            "route_file": self.route_file,
            "route_id": self.route_id,
            "scenarios": [sc.to_dict() for sc in self.scenarios],
            "keypoints": self.keypoints,
            "seed": self.seed,
        }

    # Provide mapping-like convenience
    def copy(self) -> TaskConfig:
        """Return a shallow copy (dataclasses.replace alternative)."""

        import copy

        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    #  Runtime helpers
    # ------------------------------------------------------------------

    def resolve_routes(self, map: carla.Map) -> None:
        """Populate ``ego_vehicles.routes`` with ``carla.Transform`` objects.

        This method is idempotent; calling it multiple times has no effect
        after the first successful conversion.  The yaw component is set to 0
        degrees because the XML route file stores only (x,y,z).
        """

        if self.ego_vehicles.routes.get("hero"):
            return  # already resolved

        if not self.keypoints:
            logger.warning("keypoints is empty; cannot resolve hero route")
            return

        transforms = [map.get_waypoint(carla.Location(x=x, y=y, z=z)).transform for x, y, z in self.keypoints]
        self.ego_vehicles.routes["hero"] = transforms


# ---------------------------------------------------------------------------
#  TaskSet – collection helper
# ---------------------------------------------------------------------------


class TaskSet(Sequence[TaskConfig]):
    """A collection of TaskConfig with builder utilities."""

    def __init__(self, tasks: Iterable[TaskConfig]):
        self._tasks = list(tasks)
        if not self._tasks:
            raise ValueError("TaskSet cannot be empty")

    # ----- Python collection API -----
    def __len__(self) -> int:  # noqa: D401 – short method name is fine
        return len(self._tasks)

    def __getitem__(self, index: int) -> TaskConfig:
        return self._tasks[index]

    def __iter__(self) -> Iterator[TaskConfig]:
        return iter(self._tasks)

    # ----- Builders -----
    @classmethod
    def from_route_files(
        cls,
        paths: Sequence[str | Path],
        *,
        num_npc_vehicles: int | tuple[int, int] = 0,
        num_npc_walkers: int | tuple[int, int] = 0,
        weather: str = "dynamic_1.0",
    ) -> TaskSet:
        """Build TaskSet from route XML files (CARLA Leaderboard format)."""

        tasks: list[TaskConfig] = []
        for p in paths:
            tasks.extend(
                cls._parse_route_file(
                    Path(p), weather=weather, num_npc_vehicles=num_npc_vehicles, num_npc_walkers=num_npc_walkers
                )
            )
        return cls(tasks)

    @classmethod
    def build(
        cls,
        *,
        weathers: Sequence[str] | str = "dynamic_1.0",
        map_names: Sequence[str] | str = "Town01",
        num_npc_vehicles: Sequence[int | tuple[int, int]] | int | tuple[int, int] = 0,
        num_npc_walkers: Sequence[int | tuple[int, int]] | int | tuple[int, int] = 0,
        route_file: str | None = None,
        route_id: str | None = None,
    ) -> TaskSet:
        """Create TaskSet as Cartesian product of the supplied sequences.

        Any parameter may be given as a single value or a sequence.
        A single value is treated as a length-1 sequence.
        """
        seq_weather = _as_seq(weathers)
        seq_map = _as_seq(map_names)
        seq_npc_veh = _as_seq(num_npc_vehicles)
        seq_npc_wlk = _as_seq(num_npc_walkers)

        tasks: list[TaskConfig] = []
        for w in seq_weather:
            for town in seq_map:
                for veh in seq_npc_veh:
                    for wlk in seq_npc_wlk:
                        tasks.append(
                            TaskConfig(
                                weather=w,
                                map_name=town,
                                num_npc_vehicles=veh,
                                num_npc_walkers=wlk,
                                route_file=route_file,
                                route_id=route_id,
                            )
                        )
        return cls(tasks)

    @classmethod
    def random_stream(
        cls,
        *,
        rng: np.random.Generator | None = None,
        **kwargs,
    ):
        """Yield TaskConfig indefinitely, each time with new random parameters.

        `random_kwargs` are forwarded to TaskConfig.random().
        """
        rng = rng or np.random.default_rng()
        while True:
            yield TaskConfig.sample(rng=rng, **kwargs)

    # ----- Utilities -----
    def sample_task(self, rng: np.random.Generator | None = None) -> TaskConfig:
        rng = rng or np.random.default_rng()
        return rng.choice(self._tasks)

    def to_dict(self) -> dict[int, dict[str, Any]]:
        """Return {idx: task_dict} mapping for *CarlaEnv*."""

        return {i: t.to_dict() for i, t in enumerate(self._tasks)}

    # ----- Internal -----
    @staticmethod
    def _parse_route_file(
        path: Path,
        *,
        weather: str,
        num_npc_vehicles: int | tuple[int, int],
        num_npc_walkers: int | tuple[int, int],
    ) -> list[TaskConfig]:
        """Parse one XML routes file into TaskConfig list using srunner RouteParser."""

        if not path.exists():
            raise FileNotFoundError(path)

        route_confs = RouteParser.parse_routes_file(str(path))
        task_list: list[TaskConfig] = []

        for rc in route_confs:
            scenarios = [ScenarioConfig.from_srunner(sc) for sc in rc.scenario_configs]

            # Store raw floats only (conversion deferred)
            float_pts = [(p.x, p.y, p.z) for p in rc.keypoints]
            ego_conf = EgoVehiclesConfig()

            task = TaskConfig(
                weather=weather,
                map_name=rc.town,
                num_npc_vehicles=num_npc_vehicles,
                num_npc_walkers=num_npc_walkers,
                ego_vehicles=ego_conf,
                route_file=str(path),
                route_id=rc.name.replace("RouteScenario_", ""),
                scenarios=scenarios,
                keypoints=float_pts,
            )
            task_list.append(task)

        return task_list


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------


def _normalise_range(val: int | Sequence[int]) -> int | tuple[int, int]:
    """Convert int or list-like to consistent representation."""

    if isinstance(val, Sequence) and not isinstance(val, str | bytes):
        if len(val) != 2:
            raise ValueError("Range must be length-2 [min,max] sequence")
        return int(val[0]), int(val[1])
    return int(val)


def _sample_range(val: int | tuple[int, int], rng: np.random.Generator) -> int:
    if isinstance(val, tuple):
        low, high = val
        return int(rng.integers(low, high + 1))
    return int(val)


def _auto_cast(text: str) -> Any:
    """Convert string to int/float where possible."""

    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _as_seq(val: Any) -> Any:
    """Ensure the input is a sequence (but not str / bytes)."""
    if isinstance(val, Sequence) and not isinstance(val, str | bytes):
        return val
    return (val,)
