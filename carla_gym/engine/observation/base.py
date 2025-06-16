# base class for observation managers


class BaseObservation:
    def __init__(self):
        self._expected_frame = None
        self._define_obs_space()

    def _define_obs_space(self):
        raise NotImplementedError

    def attach_ego_vehicle(self, parent_actor):
        raise NotImplementedError

    def set_expected_frame(self, frame: int | None = None) -> None:
        self._expected_frame = frame

    def get_observation(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError
