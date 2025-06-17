from __future__ import annotations

"""Random route generator inspired by CaRL Appendix D.2.

The generator produces a >= 1 km route as a list of ``carla.Transform`` s.
Current implementation is a pragmatic first version:
- Pick a random start waypoint from the map topology.
- Iteratively extend the route ~1 m at a time.
  * At intersections choose a random outgoing lane.
  * Otherwise follow lane with 10 % chance of lane change.
- Stop when accumulated distance >= ``min_len``.
- Validate that the first waypoint can spawn a vehicle; if not, resample.

The algorithm uses rejection sampling with an upper bound on trials to
avoid infinite loops.
"""


import carla
import numpy as np

from carla_gym.utils.logger import setup_logger

logger = setup_logger(__name__)

# Resolution when stepping forward along a lane (metres)
_STEP_LEN: float = 1.0


class RandomRouteGenerator:
    """Generate a random route of at least *min_len* metres."""

    def __init__(
        self,
        c_map: carla.Map,
        rng: np.random.Generator | None = None,
        *,
        min_len: float = 1000.0,
        lane_change_prob: float = 0.1,
        max_trials: int = 100,
    ) -> None:
        self._map = c_map
        self._rng = rng or np.random.default_rng()
        self._min_len = min_len
        self._lane_change_prob = lane_change_prob
        self._max_trials = max_trials

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------

    def generate(self) -> list[carla.Transform]:
        """Return a list of waypoints (as Transform) representing the route."""

        for _ in range(self._max_trials):
            try:
                route = self._try_generate_once()
            except RuntimeError:
                continue
            if self._route_length(route) >= self._min_len:
                return route
        raise RuntimeError("Failed to generate a valid route after max_trials")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _try_generate_once(self) -> list[carla.Transform]:
        # Pick a random start waypoint (non-junction preferred)
        topology = self._map.get_topology()
        start_wp, _ = self._rng.choice(topology)
        # If junction, move a bit forward until outside
        steps = 0
        while start_wp.is_junction and steps < 100:
            start_wp = start_wp.next(_STEP_LEN)[0]
            steps += 1

        route = [start_wp.transform]
        curr_wp = start_wp
        acc_len = 0.0

        while acc_len < self._min_len:
            next_wps = curr_wp.next(_STEP_LEN)
            if not next_wps:
                # Dead-end: abort this trial
                raise RuntimeError("Dead end reached")

            # Handle intersections
            if len(next_wps) > 1:
                curr_wp = self._rng.choice(next_wps)
            else:
                curr_wp = next_wps[0]
                # Random lane change on straight segments
                if self._rng.random() < self._lane_change_prob:
                    side_wp = self._pick_side_lane(curr_wp)
                    if side_wp is not None:
                        curr_wp = side_wp
            prev_tr = route[-1]
            seg_len = prev_tr.location.distance(curr_wp.transform.location)
            acc_len += seg_len
            route.append(curr_wp.transform)

            if len(route) > 4000:  # safety
                raise RuntimeError("Route too long without reaching target length")

        # Validate spawnability at first transform
        if not self._validate_spawn(route[0]):
            raise RuntimeError("Spawn validation failed")
        return route

    def _pick_side_lane(self, wp: carla.Waypoint) -> carla.Waypoint | None:
        """Return left/right lane waypoint with same direction if exists."""

        msg = (
            "FIXME: _pick_side_lane is not sufficiently implemented yet. Need to implement more robust lane change logic like: "
            "check if the side lane is a valid lane to change to, if the road has long enough to the next intersection, etc."
        )
        logger.warning(msg)

        choices: list[carla.Waypoint] = []
        left = wp.get_left_lane()
        right = wp.get_right_lane()
        for side in (left, right):
            if side is None:
                continue
            if side.lane_type == carla.LaneType.Driving and side.lane_id * wp.lane_id > 0:
                choices.append(side)
        if not choices:
            return None
        return self._rng.choice(choices)

    def _route_length(self, transforms: list[carla.Transform]) -> float:
        length = 0.0
        for i in range(len(transforms) - 1):
            length += transforms[i].location.distance(transforms[i + 1].location)
        return length

    def _validate_spawn(self, transform: carla.Transform) -> bool:
        """Check if a vehicle can be spawned at the given transform."""
        # This is a heuristic; we rely on map height validity.
        wp = self._map.get_waypoint(transform.location, project_to_road=True)
        return wp is not None
