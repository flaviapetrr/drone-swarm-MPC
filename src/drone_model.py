from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import numpy as np


class OurModel(CtrlAviary):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        self._init_done = False
        self._video_started = False

        super().__init__(*args, **kwargs)

        self._init_done = True

    def _startVideoRecording(self):
        # The parent calls this from __init__() and from every reset().
        # Skip the constructor call and any call after the first recording.
        if not self._init_done or self._video_started:
            return
        self._video_started = True
        super()._startVideoRecording()
