"""
SimpleStateActionWrapper
------------------------
Flattens selected CARLA observations into a single vector and, optionally,
exposes an ``(acc, steer)`` action interface (acc ∈ [−1, 1]).
"""

from collections.abc import Sequence

import carla
import cv2
import gymnasium as gym
import numpy as np

from carla_gym.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_NUM_NPC_VEHICLES: dict[str, int] = {
    "Town01": 120,
    "Town02": 70,
    "Town03": 70,
    "Town04": 150,
    "Town05": 120,
    "Town06": 120,
    "Town07": 100,
    "Town10H": 100,
    "Town12": 120,
    "Town13": 120,
    "Town15": 70,
}
DEFAULT_NUM_NPC_WALKERS: dict[str, int] = {
    "Town01": 120,
    "Town02": 70,
    "Town03": 70,
    "Town04": 80,
    "Town05": 120,
    "Town06": 80,
    "Town07": 100,
    "Town10H": 100,
    "Town12": 120,
    "Town13": 120,
    "Town15": 70,
}

_OBS_EXTRACTORS = {
    "speed": lambda o: o["speed"]["speed_xy"],
    "speed_limit": lambda o: o["control"]["speed_limit"],
    "control": lambda o: np.array(
        [
            o["control"]["throttle"],
            o["control"]["steer"],
            o["control"]["brake"],
            o["control"]["gear"] / 5.0,
        ],
        dtype=np.float32,
    ),
    "acc_xy": lambda o: o["velocity"]["acc_xy"],
    "vel_xy": lambda o: o["velocity"]["vel_xy"],
    "vel_ang_z": lambda o: o["velocity"]["vel_ang_z"],
}


def _extract_space(space: gym.spaces.Dict, key: str) -> gym.spaces.Box:  # type: ignore[override]
    if key in space:
        sub = space[key]
        if isinstance(sub, gym.spaces.Box):
            return sub
        raise ValueError(f"'{key}' should be Box, got {type(sub).__name__}.")
    raise KeyError(key)


def _fmt(arr) -> str:
    return (
        np.array2string(np.asarray(arr), precision=2, separator=",", suppress_small=True) if arr is not None else "N/A"
    )


def _put(img: np.ndarray, txt: str, org: tuple[int, int], scale: float):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def _font(target: int, it: int = 10) -> float:
    s = 1.0
    for _ in range(it):
        h = cv2.getTextSize("Ay", cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][1]
        if abs(h - target) < 1:
            break
        s *= target / h
    return s


class SimpleStateActionWrapper(gym.Wrapper):
    """Flatten observation dict and remap action space."""

    def __init__(
        self,
        env: gym.Env,
        *,
        input_states: Sequence[str] | None = None,
        acc_as_action: bool = True,
    ) -> None:
        super().__init__(env)
        self.input_states: tuple[str, ...] = tuple(input_states or ("control", "vel_xy"))
        self.acc_as_action: bool = acc_as_action
        self.eval_mode: bool = False

        low, high = self._state_bounds()

        observation_space = {}
        observation_space["state"] = gym.spaces.Box(low, high, dtype=np.float32)
        if "birdview" in self.observation_space.keys():
            observation_space["birdview"] = self.observation_space["birdview"]["masks"]
        if "front_rgb" in self.observation_space.keys():
            observation_space["front_rgb"] = self.observation_space["front_rgb"]["data"]
        self.observation_space = gym.spaces.Dict(observation_space)

        if self.acc_as_action:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self._render: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        # self.env.unwrapped.set_task_idx(np.random.choice(self.env.unwrapped.num_tasks))
        self._setup_eval_population()
        obs, info = self.env.reset(seed=seed, options=options)
        for ctrl in (
            carla.VehicleControl(manual_gear_shift=True, gear=1),
            carla.VehicleControl(manual_gear_shift=False),
        ):
            obs, *_ = self.env.step(ctrl)
        self.env.unwrapped._update_timestamp(reset_called=True)
        proc = self._proc_obs(obs, train=True)
        self._render.update(
            {
                "timestamp": self.env.unwrapped.timestamp,
                "prev_obs": proc,
                "prev_im_render": obs["birdview"]["rendered"],
                "reward_debug": info["info"]["reward_debug"],
                "terminal_debug": info["info"]["terminal_debug"],
            }
        )
        return proc, {}

    def step(self, action):  # type: ignore[override]
        ctrl = self._proc_act(action, train=True)
        obs, reward, terminated, truncated, info = self.env.step(ctrl)
        proc = self._proc_obs(obs, train=True)
        self._render = {
            "timestamp": self.env.unwrapped.timestamp,
            "obs": self._render["prev_obs"],
            "prev_obs": proc,
            "im_render": self._render["prev_im_render"],
            "prev_im_render": obs["birdview"]["rendered"],
            "action": ctrl,
            "reward_debug": info["reward_debug"],
            "terminal_debug": info["terminal_debug"],
        }
        return proc, reward, terminated, truncated, info

    def render(self) -> np.ndarray:  # noqa: D401
        for k in ("action_value", "action_log_probs", "action_mu", "action_sigma"):
            self._render[k] = getattr(self, k, None)
        return self._draw(self._render)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        spaces = []
        if "speed" in self.input_states:
            spaces.append(self.observation_space["speed"]["speed_xy"])
        if "speed_limit" in self.input_states:
            spaces.append(self.observation_space["control"]["speed_limit"])
        if "control" in self.input_states:
            for sub in ("throttle", "steer", "brake", "gear"):
                spaces.append(self.observation_space["control"][sub])
        if "acc_xy" in self.input_states:
            spaces.append(self.observation_space["velocity"]["acc_xy"])
        if "vel_xy" in self.input_states:
            spaces.append(self.observation_space["velocity"]["vel_xy"])
        if "vel_ang_z" in self.input_states:
            spaces.append(self.observation_space["velocity"]["vel_ang_z"])
        low = np.concatenate([s.low for s in spaces])
        high = np.concatenate([s.high for s in spaces])
        return low, high

    def _proc_obs(self, obs: dict, *, train: bool) -> dict:
        processed_obs = {}
        obs_keys = list(self.observation_space.keys())
        if "state" in obs_keys:
            vec = np.concatenate(
                [_OBS_EXTRACTORS[k](obs).flatten() for k in self.input_states if k != "birdview" and k != "front_rgb"]
            ).astype(np.float32)
            processed_obs["state"] = vec
        if "birdview" in obs_keys:
            bev = obs["birdview"]["masks"]
            processed_obs["birdview"] = bev
        if "front_rgb" in obs_keys:
            processed_obs["front_rgb"] = obs["front_rgb"]["data"]
        return processed_obs

    def _proc_act(self, act, *, train: bool):
        if not train:
            act = act[0]
        act = act.astype(np.float32)
        if self.acc_as_action:
            acc, steer = act
            throttle = max(acc, 0.0)
            brake = max(-acc, 0.0)
        else:
            throttle, steer, brake = act
        return carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )

    def _setup_eval_population(self):
        task = self.env.unwrapped._task
        handler = self.env.unwrapped._ev_handler
        if self.eval_mode:
            m = self.env.unwrapped.carla_map
            task["num_npc_vehicles"] = DEFAULT_NUM_NPC_VEHICLES[m]
            task["num_npc_walkers"] = DEFAULT_NUM_NPC_WALKERS[m]
        for ev_id in handler._terminal_configs:
            handler._terminal_configs[ev_id]["kwargs"]["eval_mode"] = self.eval_mode

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw(r: dict, *, fscale: float | None = None) -> np.ndarray:
        left = r.get("im_render", r["prev_im_render"])
        h, w, _ = left.shape
        canvas = np.zeros((h, w * 2, 3), np.uint8)
        canvas[:, :w] = left
        if fscale is None:
            fscale = _font(int(h * 0.03))
        th = cv2.getTextSize("Ay", cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)[0][1]
        lh = int(th * 1.8)
        y, yr, xr = th + 3, th + 3, w
        _put(canvas, f"step:{r['timestamp']['step']:5}, frame:{r['timestamp']['frame']:5}", (3, y), fscale)
        y += lh
        if r.get("action_value") is not None and r.get("action_log_probs") is not None:
            _put(
                canvas,
                f"a{_fmt(r.get('action'))} v:{r['action_value']:.2f} p:{r['action_log_probs']:.2f}",
                (3, y),
                fscale,
            )
            y += lh
        _put(canvas, f"s{_fmt(r.get('obs'))}", (3, y), fscale)
        _put(canvas, f"a{_fmt(r.get('action_mu'))} b{_fmt(r.get('action_sigma'))}", (xr, yr), fscale)
        yr += lh
        for t in r["reward_debug"]["debug_texts"] + r["terminal_debug"]["debug_texts"]:
            _put(canvas, t, (xr, yr), fscale)
            yr += lh
        return canvas
