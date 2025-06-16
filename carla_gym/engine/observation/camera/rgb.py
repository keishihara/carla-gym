import copy
import weakref
from queue import Empty, Queue

import carla
import numpy as np
from gymnasium import spaces

from carla_gym.engine.observation.base import BaseObservation


class CameraRGB(BaseObservation):
    """
    Template configs:
    obs_configs = {
        "module": "camera.rgb",
        "location": [-5.5, 0, 2.8],
        "rotation": [0, -15, 0],
        "frame_stack": 1,
        "width": 1920,
        "height": 1080
    }
    frame_stack: [Image(t-2), Image(t-1), Image(t)]
    """

    def __init__(self, obs_configs):
        self._sensor_type = "sensor.camera.rgb"

        self._height = obs_configs["height"]
        self._width = obs_configs["width"]
        self._fov = obs_configs["fov"]
        self._channels = 3

        location = carla.Location(
            x=float(obs_configs["x"]),
            y=float(obs_configs["y"]),
            z=float(obs_configs["z"]),
        )
        rotation = carla.Rotation(
            roll=float(obs_configs["roll"]),
            pitch=float(obs_configs["pitch"]),
            yaw=float(obs_configs["yaw"]),
        )

        self._camera_transform = carla.Transform(location, rotation)

        self._sensor = None
        self._queue_timeout = 10.0
        self._image_queue = None

        super().__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {
                "frame": spaces.Discrete(2**32 - 1),
                "data": spaces.Box(low=0, high=255, shape=(self._height, self._width, self._channels), dtype=np.uint8),
            }
        )

    def attach_ego_vehicle(self, parent_actor):
        init_obs = np.zeros([self._height, self._width, self._channels], dtype=np.uint8)
        self._image_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find(self._sensor_type)
        bp.set_attribute("image_size_x", str(self._width))
        bp.set_attribute("image_size_y", str(self._height))
        bp.set_attribute("fov", str(self._fov))

        self._sensor = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda image: self._parse_image(weak_self, image))

    def get_observation_old(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1

        try:
            frame, data = self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception("RGB sensor took too long!") from None

        return {"frame": frame, "data": data}

    def get_observation(self):
        """
        Return the RGB image strictly matching the expected frame (if one was set by ObsManagerHandler).
        Older frames are discarded; newer frames are waited for.
        """
        expected = getattr(self, "_expected_frame", None)
        if expected is None:
            expected = self._world.get_snapshot().frame

        assert self._image_queue.qsize() <= 1

        while True:
            # if sensor has pushed more than one frame, drop the oldest one
            while self._image_queue.qsize() > 1:
                _ = self._image_queue.get_nowait()

            try:
                frame, data = self._image_queue.get(True, self._queue_timeout)
            except Empty:
                raise RuntimeError("RGB sensor took too long!") from None

            if frame < expected:
                # older frame, drop it and wait for the next frame
                continue

            if frame > expected:
                # it is likely that we missed the expected frame. wait for the next frame
                continue

            return {"frame": frame, "data": data}

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        self._sensor = None
        self._world = None
        self._image_queue = None

    @staticmethod
    def _parse_image(weak_self, carla_image):
        self = weak_self()

        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        np_img = copy.deepcopy(np_img)
        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3]
        np_img = np_img[:, :, ::-1]
        self._image_queue.put((carla_image.frame, np_img))
