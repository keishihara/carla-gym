from __future__ import annotations

import carla
import cv2
import gymnasium as gym
import numpy as np

from carla_gym.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_NUM_NPC_VEHICLES = {
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
DEFAULT_NUM_NPC_WALKERS = {
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


class SimpleStateActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, input_states: list[str] | None = None, acc_as_action: bool = True):
        super().__init__(env)
        self._input_states = input_states or ["control", "vel_xy"]
        self._acc_as_action = acc_as_action
        self._render_dict = {}

        state_spaces: list[gym.spaces.Box] = []
        if "speed" in self._input_states:
            state_spaces.append(self.observation_space["speed"]["speed_xy"])
        if "speed_limit" in self._input_states:
            state_spaces.append(self.observation_space["control"]["speed_limit"])
        if "control" in self._input_states:
            state_spaces.append(self.observation_space["control"]["throttle"])
            state_spaces.append(self.observation_space["control"]["steer"])
            state_spaces.append(self.observation_space["control"]["brake"])
            state_spaces.append(self.observation_space["control"]["gear"])
        if "acc_xy" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["acc_xy"])
        if "vel_xy" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["vel_xy"])
        if "vel_ang_z" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["vel_ang_z"])

        state_low = np.concatenate([s.low for s in state_spaces])
        state_high = np.concatenate([s.high for s in state_spaces])

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
                "birdview": self.observation_space["birdview"]["masks"],
            }
        )

        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            self.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        self.eval_mode = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        self.env.unwrapped.set_task_idx(np.random.choice(self.env.unwrapped.num_tasks))
        if self.eval_mode:
            self.env.unwrapped._task["num_npc_vehicles"] = DEFAULT_NUM_NPC_VEHICLES[self.env.unwrapped.carla_map]
            self.env.unwrapped._task["num_npc_walkers"] = DEFAULT_NUM_NPC_WALKERS[self.env.unwrapped.carla_map]
            for ev_id in self.env.unwrapped._ev_handler._terminal_configs:
                self.env.unwrapped._ev_handler._terminal_configs[ev_id]["kwargs"]["eval_mode"] = True
        else:
            for ev_id in self.env.unwrapped._ev_handler._terminal_configs:
                self.env.unwrapped._ev_handler._terminal_configs[ev_id]["kwargs"]["eval_mode"] = False

        obs, info = self.env.reset(seed=seed, options=options)
        action = carla.VehicleControl(manual_gear_shift=True, gear=1)
        obs, *_ = self.env.step(action)
        action = carla.VehicleControl(manual_gear_shift=False)
        obs, *_ = self.env.step(action)

        self.env.unwrapped._update_timestamp(reset_called=True)

        obs_processed = self.process_obs(obs, self._input_states)
        self._render_dict.update(
            {
                "timestamp": self.env.unwrapped.timestamp,
                "prev_obs": obs_processed,
                "prev_im_render": obs["birdview"]["rendered"],
                "reward_debug": info["info"]["reward_debug"],
                "terminal_debug": info["info"]["terminal_debug"],
            }
        )
        return obs_processed, {}

    def step(self, action: dict) -> tuple:
        action = self.process_act(action, self._acc_as_action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._render_dict = {
            "timestamp": self.env.unwrapped.timestamp,
            "obs": self._render_dict["prev_obs"],
            "prev_obs": obs,
            "im_render": self._render_dict["prev_im_render"],
            "prev_im_render": obs["birdview"]["rendered"],
            "action": action,
            "reward_debug": info["reward_debug"],
            "terminal_debug": info["terminal_debug"],
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        train render: used in train_rl.py
        """
        self._render_dict["action_value"] = getattr(self, "action_value", None)
        self._render_dict["action_log_probs"] = getattr(self, "action_log_probs", None)
        self._render_dict["action_mu"] = getattr(self, "action_mu", None)
        self._render_dict["action_sigma"] = getattr(self, "action_sigma", None)
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict: dict, font_scale: float = 0.3) -> np.ndarray:
        im_birdview = render_dict.get("im_render", None)
        if im_birdview is None:
            im_birdview = render_dict["prev_im_render"]
        h, w, c = im_birdview.shape
        im = np.zeros([h, w * 2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        def _get_font_scale_for_height(
            target_height_pixels: int,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            thickness=1,
        ) -> float:
            test_text = "Ay"
            font_scale = 1.0
            for _ in range(10):
                (_, height), _ = cv2.getTextSize(test_text, font_face, font_scale, thickness)
                if abs(height - target_height_pixels) < 1:
                    break
                font_scale *= target_height_pixels / height
            return font_scale

        font_scale = _get_font_scale_for_height(int(h * 0.03))

        test_text = "Ay"
        (_, text_height), baseline = cv2.getTextSize(test_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        line_spacing = int(text_height * 1.8)

        def _get_string(render_dict, key):
            if key in render_dict and isinstance(render_dict[key], np.ndarray):
                return np.array2string(render_dict[key], precision=2, separator=",", suppress_small=True)
            return "N/A"

        action_str = _get_string(render_dict, "action")
        mu_str = _get_string(render_dict, "action_mu")
        sigma_str = _get_string(render_dict, "action_sigma")
        state_str = _get_string(render_dict, "obs")

        current_y = text_height + 3

        txt_t = f"step:{render_dict['timestamp']['step']:5}, frame:{render_dict['timestamp']['frame']:5}"
        im = cv2.putText(im, txt_t, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        current_y += line_spacing

        if (
            "action_value" in render_dict
            and render_dict["action_value"] is not None
            and "action_log_probs" in render_dict
            and render_dict["action_log_probs"] is not None
        ):
            txt_1 = f"a{action_str} v:{render_dict['action_value']:5.2f} p:{render_dict['action_log_probs']:5.2f}"
            im = cv2.putText(im, txt_1, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            current_y += line_spacing

        txt_2 = f"s{state_str}"
        im = cv2.putText(im, txt_2, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        current_y_right = text_height + 3

        txt_3 = f"a{mu_str} b{sigma_str}"
        im = cv2.putText(im, txt_3, (w, current_y_right), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        current_y_right += line_spacing

        for txt in render_dict["reward_debug"]["debug_texts"] + render_dict["terminal_debug"]["debug_texts"]:
            im = cv2.putText(im, txt, (w, current_y_right), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            current_y_right += line_spacing

        return im

    @staticmethod
    def process_obs(obs: dict, input_states: list[str], train: bool = True) -> dict:
        state_list = []
        if "speed" in input_states:
            state_list.append(obs["speed"]["speed_xy"])
        if "speed_limit" in input_states:
            state_list.append(obs["control"]["speed_limit"])
        if "control" in input_states:
            state_list.append(obs["control"]["throttle"])
            state_list.append(obs["control"]["steer"])
            state_list.append(obs["control"]["brake"])
            state_list.append(obs["control"]["gear"] / 5.0)
        if "acc_xy" in input_states:
            state_list.append(obs["velocity"]["acc_xy"])
        if "vel_xy" in input_states:
            state_list.append(obs["velocity"]["vel_xy"])
        if "vel_ang_z" in input_states:
            state_list.append(obs["velocity"]["vel_ang_z"])

        state = np.concatenate(state_list)

        birdview = obs["birdview"]["masks"]

        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {"state": state.astype(np.float32), "birdview": birdview}
        return obs_dict

    @staticmethod
    def process_act(action: dict, acc_as_action: bool, train: bool = True) -> dict:
        if not train:
            action = action[0]
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control


class SimpleStateActionWrapperNew(gym.Wrapper):
    def __init__(self, env: gym.Env, input_states: list[str] | None = None, acc_as_action: bool = True):
        super().__init__(env)

        self._input_states = input_states or ["control", "vel_xy"]
        self._acc_as_action = acc_as_action
        self._render_dict = {}

        state_spaces: list[gym.spaces.Box] = []
        if "speed" in self._input_states:
            state_spaces.append(self.observation_space["speed"]["speed_xy"])
        if "speed_limit" in self._input_states:
            state_spaces.append(self.observation_space["control"]["speed_limit"])
        if "control" in self._input_states:
            state_spaces.append(self.observation_space["control"]["throttle"])
            state_spaces.append(self.observation_space["control"]["steer"])
            state_spaces.append(self.observation_space["control"]["brake"])
            state_spaces.append(self.observation_space["control"]["gear"])
        if "acc_xy" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["acc_xy"])
        if "vel_xy" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["vel_xy"])
        if "vel_ang_z" in self._input_states:
            state_spaces.append(self.observation_space["velocity"]["vel_ang_z"])

        state_low = np.concatenate([s.low for s in state_spaces])
        state_high = np.concatenate([s.high for s in state_spaces])

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
                "birdview": self.observation_space["birdview"]["masks"],
            }
        )

        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            self.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        self.eval_mode = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        self.env.unwrapped.set_task_idx(np.random.choice(self.env.unwrapped.num_tasks))
        if self.eval_mode:
            self.env.unwrapped._task["num_npc_vehicles"] = DEFAULT_NUM_NPC_VEHICLES[self.env.unwrapped.carla_map]
            self.env.unwrapped._task["num_npc_walkers"] = DEFAULT_NUM_NPC_WALKERS[self.env.unwrapped.carla_map]
            for ev_id in self.env.unwrapped._ev_handler._terminal_configs:
                self.env.unwrapped._ev_handler._terminal_configs[ev_id]["kwargs"]["eval_mode"] = True
        else:
            for ev_id in self.env.unwrapped._ev_handler._terminal_configs:
                self.env.unwrapped._ev_handler._terminal_configs[ev_id]["kwargs"]["eval_mode"] = False

        obs, info = self.env.reset(seed=seed, options=options)
        action = carla.VehicleControl(manual_gear_shift=True, gear=1)
        obs, *_ = self.env.step(action)
        action = carla.VehicleControl(manual_gear_shift=False)
        obs, *_ = self.env.step(action)

        self.env.unwrapped._update_timestamp(reset_called=True)
        obs_processed = self.process_obs(obs, self._input_states)
        self._render_dict.update(
            {
                "timestamp": self.env.unwrapped.timestamp,
                "prev_obs": obs_processed,
                "prev_im_render": obs["birdview"]["rendered"],
                "reward_debug": info["info"]["reward_debug"],
                "terminal_debug": info["info"]["terminal_debug"],
            }
        )
        return obs_processed, {}

    def step(self, action: dict) -> tuple:
        action = self.process_act(action, self._acc_as_action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._render_dict = {
            "timestamp": self.env.unwrapped.timestamp,
            "obs": self._render_dict["prev_obs"],
            "prev_obs": obs,
            "im_render": self._render_dict["prev_im_render"],
            "prev_im_render": obs["birdview"]["rendered"],
            "action": action,
            "reward_debug": info["reward_debug"],
            "terminal_debug": info["terminal_debug"],
        }
        return obs, reward, terminated, truncated, info

    @staticmethod
    def process_obs(obs: dict, input_states: list[str], train: bool = True) -> dict:
        state_list = []
        if "speed" in input_states:
            state_list.append(obs["speed"]["speed_xy"])
        if "speed_limit" in input_states:
            state_list.append(obs["control"]["speed_limit"])
        if "control" in input_states:
            state_list.append(obs["control"]["throttle"])
            state_list.append(obs["control"]["steer"])
            state_list.append(obs["control"]["brake"])
            state_list.append(obs["control"]["gear"] / 5.0)
        if "acc_xy" in input_states:
            state_list.append(obs["velocity"]["acc_xy"])
        if "vel_xy" in input_states:
            state_list.append(obs["velocity"]["vel_xy"])
        if "vel_ang_z" in input_states:
            state_list.append(obs["velocity"]["vel_ang_z"])

        state = np.concatenate(state_list)

        birdview = obs["birdview"]["masks"]

        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {"state": state.astype(np.float32), "birdview": birdview}
        return obs_dict

    @staticmethod
    def process_act(action: dict, acc_as_action: bool, train: bool = True) -> dict:
        if not train:
            action = action[0]
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control


class RenderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, input_states: list[str] | None = None, acc_as_action: bool = True):
        super().__init__(env)
        self._render_dict = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        obs, info = self.env.reset(seed=seed, options=options)

        original_obs = self.env.unwrapped._get_observation()["hero"]

        self._render_dict.update(
            {
                "timestamp": self.env.unwrapped.timestamp,
                "prev_obs": obs,
                "prev_im_render": original_obs["birdview"]["rendered"],
                "reward_debug": info.get("info", {}).get("reward_debug", {"debug_texts": []}),
                "terminal_debug": info.get("info", {}).get("terminal_debug", {"debug_texts": []}),
            }
        )
        return obs, info

    def step(self, action: dict) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        original_obs = self.env.unwrapped._get_observation()["hero"]
        self._render_dict = {
            "timestamp": self.env.unwrapped.timestamp,
            "obs": self._render_dict.get("prev_obs", obs),
            "prev_obs": obs,
            "im_render": self._render_dict.get("prev_im_render"),
            "prev_im_render": original_obs["birdview"]["rendered"],
            "action": action,
            "reward_debug": info.get("reward_debug", {"debug_texts": []}),
            "terminal_debug": info.get("terminal_debug", {"debug_texts": []}),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        train render: used in train_rl.py
        """
        self._render_dict["action_value"] = getattr(self, "action_value", None)
        self._render_dict["action_log_probs"] = getattr(self, "action_log_probs", None)
        self._render_dict["action_mu"] = getattr(self, "action_mu", None)
        self._render_dict["action_sigma"] = getattr(self, "action_sigma", None)
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict: dict, font_scale: float = 0.3) -> np.ndarray:
        im_birdview = render_dict.get("im_render", None)
        if im_birdview is None:
            im_birdview = render_dict["prev_im_render"]
        h, w, c = im_birdview.shape
        im = np.zeros([h, w * 2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        def _get_font_scale_for_height(
            target_height_pixels: int,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            thickness=1,
        ) -> float:
            test_text = "Ay"
            font_scale = 1.0
            for _ in range(10):
                (_, height), _ = cv2.getTextSize(test_text, font_face, font_scale, thickness)
                if abs(height - target_height_pixels) < 1:
                    break
                font_scale *= target_height_pixels / height
            return font_scale

        font_scale = _get_font_scale_for_height(int(h * 0.03))

        test_text = "Ay"
        (_, text_height), baseline = cv2.getTextSize(test_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        line_spacing = int(text_height * 1.8)

        def _get_string(render_dict, key):
            if key in render_dict and isinstance(render_dict[key], np.ndarray):
                return np.array2string(render_dict[key], precision=2, separator=",", suppress_small=True)
            return "N/A"

        action_str = _get_string(render_dict, "action")
        mu_str = _get_string(render_dict, "action_mu")
        sigma_str = _get_string(render_dict, "action_sigma")
        state_str = _get_string(render_dict, "obs")

        current_y = text_height + 3

        txt_t = f"step:{render_dict['timestamp']['step']:5}, frame:{render_dict['timestamp']['frame']:5}"
        im = cv2.putText(im, txt_t, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        current_y += line_spacing

        if (
            "action_value" in render_dict
            and render_dict["action_value"] is not None
            and "action_log_probs" in render_dict
            and render_dict["action_log_probs"] is not None
        ):
            txt_1 = f"a{action_str} v:{render_dict['action_value']:5.2f} p:{render_dict['action_log_probs']:5.2f}"
            im = cv2.putText(im, txt_1, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            current_y += line_spacing

        txt_2 = f"s{state_str}"
        im = cv2.putText(im, txt_2, (3, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        current_y_right = text_height + 3

        txt_3 = f"a{mu_str} b{sigma_str}"
        im = cv2.putText(im, txt_3, (w, current_y_right), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        current_y_right += line_spacing

        for txt in render_dict["reward_debug"]["debug_texts"] + render_dict["terminal_debug"]["debug_texts"]:
            im = cv2.putText(im, txt, (w, current_y_right), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            current_y_right += line_spacing

        return im
