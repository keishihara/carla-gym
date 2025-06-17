"""Tests for Leaderboard 2.0 integration in CarlaGym.

This module tests the integration of leaderboard_2_0 scenarios,
including XML parsing, route loading, and TaskConfig integration.
"""

import pytest
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

from carla_gym.envs.leaderboard_scenario_loader import (
    LeaderboardScenarioLoader,
    LeaderboardRoute,
    create_leaderboard_loader,
)
from carla_gym.envs.task_config import TaskConfig


class TestLeaderboardScenarioLoader:
    """Test suite for LeaderboardScenarioLoader."""

    @pytest.fixture
    def sample_xml_content(self):
        """Sample XML content for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="test_route" town="Town13">
        <weathers>
            <weather route_percentage="0" cloudiness="5.0" precipitation="0.0" 
                    precipitation_deposits="0.0" wetness="0.0" wind_intensity="10.0" 
                    sun_azimuth_angle="-1.0" sun_altitude_angle="90.0" fog_density="2.0"/>
        </weathers>
        <waypoints>
            <position x="100.0" y="200.0" z="1.0"/>
            <position x="150.0" y="220.0" z="1.0"/>
            <position x="200.0" y="240.0" z="1.0"/>
        </waypoints>
        <scenarios>
            <scenario name="TestAccident" type="Accident">
                <trigger_point x="120.0" y="210.0" z="1.0" yaw="90"/>
                <distance value="50"/>
                <speed value="30"/>
            </scenario>
        </scenarios>
    </route>
</routes>"""

    @pytest.fixture
    def temp_leaderboard_structure(self, sample_xml_content):
        """Create temporary leaderboard 2.0 directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create directory structure
            town_dir = base_path / "50x36_town_13"
            town_dir.mkdir()

            accident_dir = town_dir / "Accident"
            accident_dir.mkdir()

            # Create sample XML files
            xml_file1 = accident_dir / "1044_0.xml"
            xml_file1.write_text(sample_xml_content)

            xml_file2 = accident_dir / "1044_1.xml"
            xml_file2.write_text(sample_xml_content.replace('id="test_route"', 'id="test_route_2"'))

            # Create another scenario type
            blocked_dir = town_dir / "BlockedIntersection"
            blocked_dir.mkdir()

            xml_file3 = blocked_dir / "1016_0.xml"
            xml_file3.write_text(
                sample_xml_content.replace('type="Accident"', 'type="BlockedIntersection"').replace(
                    'id="test_route"', 'id="blocked_route"'
                )
            )

            yield str(base_path)

    def test_initialization(self, temp_leaderboard_structure):
        """Test LeaderboardScenarioLoader initialization."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        assert loader.data_path == Path(temp_leaderboard_structure)
        assert len(loader._route_cache) > 0

    def test_scan_routes(self, temp_leaderboard_structure):
        """Test route scanning functionality."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        # Check that routes were discovered
        assert "Town13_Accident" in loader._route_cache
        assert "Town13_BlockedIntersection" in loader._route_cache

        # Check route counts
        accident_routes = loader._route_cache["Town13_Accident"]
        assert len(accident_routes) == 2  # 1044_0.xml and 1044_1.xml

        blocked_routes = loader._route_cache["Town13_BlockedIntersection"]
        assert len(blocked_routes) == 1  # 1016_0.xml

    def test_get_available_scenario_types(self, temp_leaderboard_structure):
        """Test getting available scenario types."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        scenario_types = loader.get_available_scenario_types()
        assert "Accident" in scenario_types
        assert "BlockedIntersection" in scenario_types

    def test_get_available_towns(self, temp_leaderboard_structure):
        """Test getting available towns."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        towns = loader.get_available_towns()
        assert "Town13" in towns

    def test_get_routes_by_type(self, temp_leaderboard_structure):
        """Test filtering routes by scenario type."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        accident_routes = loader.get_routes_by_type("Accident")
        assert len(accident_routes) == 2
        assert all(route.scenario_type == "Accident" for route in accident_routes)

        blocked_routes = loader.get_routes_by_type("BlockedIntersection")
        assert len(blocked_routes) == 1
        assert blocked_routes[0].scenario_type == "BlockedIntersection"

    def test_get_random_route(self, temp_leaderboard_structure):
        """Test getting random routes."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        # Test getting random route by type
        accident_route = loader.get_random_route("Accident")
        assert accident_route is not None
        assert accident_route.scenario_type == "Accident"

        # Test getting random route by town
        town_route = loader.get_random_route(town="Town13")
        assert town_route is not None
        assert town_route.town == "Town13"

    def test_create_task_config(self, temp_leaderboard_structure):
        """Test creating TaskConfig from LeaderboardRoute."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        route = loader.get_random_route("Accident")
        assert route is not None

        task_config = loader.create_task_config(route=route, num_npc_vehicles=5, num_npc_walkers=3, weather="ClearNoon")

        assert isinstance(task_config, TaskConfig)
        assert task_config.map_name == route.town
        assert task_config.num_npc_vehicles == 5
        assert task_config.num_npc_walkers == 3
        assert task_config.weather == "ClearNoon"
        assert task_config.route_file == route.xml_file
        assert task_config.route_id == route.route_id

    def test_parse_xml_route(self, temp_leaderboard_structure, sample_xml_content):
        """Test XML route parsing."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        # Create a temporary XML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(sample_xml_content)
            temp_xml = f.name

        try:
            route = loader._parse_xml_route(Path(temp_xml), "Town13", "Accident")

            assert route is not None
            assert route.route_id == "test_route"
            assert route.town == "Town13"
            assert route.scenario_type == "Accident"
            assert len(route.waypoints) == 3
            assert route.waypoints[0] == (100.0, 200.0, 1.0)
            assert len(route.scenarios) == 1
            assert route.scenarios[0].name == "TestAccident"
            assert route.scenarios[0].type == "Accident"

        finally:
            os.unlink(temp_xml)

    def test_get_stats(self, temp_leaderboard_structure):
        """Test getting loader statistics."""
        loader = LeaderboardScenarioLoader(temp_leaderboard_structure)

        stats = loader.get_stats()

        assert "total_routes" in stats
        assert "scenario_types" in stats
        assert "towns" in stats
        assert "Accident_count" in stats
        assert "BlockedIntersection_count" in stats

        assert stats["total_routes"] == 3  # 2 Accident + 1 BlockedIntersection
        assert stats["Accident_count"] == 2
        assert stats["BlockedIntersection_count"] == 1


class TestTaskConfigLeaderboardIntegration:
    """Test TaskConfig integration with Leaderboard 2.0."""

    @pytest.fixture
    def temp_leaderboard_data(self):
        """Create temporary leaderboard data for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create minimal structure
            town_dir = base_path / "50x36_town_13"
            town_dir.mkdir()

            accident_dir = town_dir / "Accident"
            accident_dir.mkdir()

            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="test_integration" town="Town13">
        <waypoints>
            <position x="100.0" y="200.0" z="1.0"/>
            <position x="200.0" y="300.0" z="1.0"/>
        </waypoints>
        <scenarios>
            <scenario name="IntegrationTest" type="Accident">
                <trigger_point x="150.0" y="250.0" z="1.0" yaw="0"/>
            </scenario>
        </scenarios>
    </route>
</routes>"""

            xml_file = accident_dir / "integration_test.xml"
            xml_file.write_text(xml_content)

            yield str(base_path)

    @patch("carla_gym.envs.task_config.create_leaderboard_loader")
    def test_from_leaderboard_2_0(self, mock_create_loader, temp_leaderboard_data):
        """Test TaskConfig.from_leaderboard_2_0 method."""
        # Mock the loader
        mock_loader = Mock()
        mock_route = Mock()
        mock_route.town = "Town13"
        mock_route.xml_file = "test.xml"
        mock_route.route_id = "test_id"

        mock_loader.get_random_route.return_value = mock_route
        mock_loader.create_task_config.return_value = TaskConfig(
            weather="ClearNoon", map_name="Town13", num_npc_vehicles=5, num_npc_walkers=3
        )
        mock_create_loader.return_value = mock_loader

        # Test the method
        task_config = TaskConfig.from_leaderboard_2_0(
            scenario_type="Accident", town="Town13", num_npc_vehicles=5, num_npc_walkers=3
        )

        assert task_config is not None
        assert task_config.map_name == "Town13"
        assert task_config.num_npc_vehicles == 5
        mock_create_loader.assert_called_once()
        mock_loader.get_random_route.assert_called_once_with("Accident", "Town13")

    @patch("carla_gym.envs.task_config.create_leaderboard_loader")
    def test_sample_leaderboard_scenarios(self, mock_create_loader):
        """Test TaskConfig.sample_leaderboard_scenarios method."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.get_available_scenario_types.return_value = ["Accident", "BlockedIntersection"]
        mock_loader.get_available_towns.return_value = ["Town13"]

        # Mock successful task creation
        def mock_from_leaderboard(*args, **kwargs):
            return TaskConfig(weather="ClearNoon", map_name="Town13", num_npc_vehicles=1, num_npc_walkers=1)

        mock_create_loader.return_value = mock_loader

        with patch.object(TaskConfig, "from_leaderboard_2_0", side_effect=mock_from_leaderboard):
            tasks = TaskConfig.sample_leaderboard_scenarios(n=3, scenario_types=["Accident"], towns=["Town13"])

        assert len(tasks) == 3
        assert all(isinstance(task, TaskConfig) for task in tasks)
        mock_create_loader.assert_called_once()


class TestCreateLeaderboardLoader:
    """Test the factory function for creating LeaderboardScenarioLoader."""

    def test_create_with_explicit_path(self, temp_leaderboard_structure):
        """Test creating loader with explicit path."""
        loader = create_leaderboard_loader(temp_leaderboard_structure)

        assert isinstance(loader, LeaderboardScenarioLoader)
        assert str(loader.data_path) == temp_leaderboard_structure

    def test_create_with_nonexistent_path(self):
        """Test creating loader with nonexistent path."""
        with pytest.raises(FileNotFoundError):
            create_leaderboard_loader("/nonexistent/path")

    @patch("os.path.exists")
    def test_auto_detect_path(self, mock_exists):
        """Test auto-detection of leaderboard path."""

        # Mock path detection
        def side_effect(path):
            return path == "carla_benchmarks/leaderboard_2_0/data"

        mock_exists.side_effect = side_effect

        with patch.object(LeaderboardScenarioLoader, "__init__", return_value=None):
            create_leaderboard_loader(None)

        # Verify that os.path.exists was called with expected paths
        expected_calls = [
            "carla_benchmarks/leaderboard_2_0/data",
            "../carla_benchmarks/leaderboard_2_0/data",
            "../../carla_benchmarks/leaderboard_2_0/data",
        ]

        for expected_path in expected_calls:
            assert any(call.args[0] == expected_path for call in mock_exists.call_args_list)


class TestLeaderboardRoute:
    """Test LeaderboardRoute dataclass."""

    def test_leaderboard_route_creation(self):
        """Test creating LeaderboardRoute instance."""
        route = LeaderboardRoute(
            route_id="test_route",
            scenario_type="Accident",
            town="Town13",
            xml_file="/path/to/route.xml",
            waypoints=[(100.0, 200.0, 1.0), (200.0, 300.0, 1.0)],
            scenarios=[],
            weather={"cloudiness": "5.0"},
        )

        assert route.route_id == "test_route"
        assert route.scenario_type == "Accident"
        assert route.town == "Town13"
        assert route.xml_file == "/path/to/route.xml"
        assert len(route.waypoints) == 2
        assert route.weather["cloudiness"] == "5.0"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
