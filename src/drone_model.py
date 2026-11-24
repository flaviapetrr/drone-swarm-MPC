from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import numpy as np


class OurModel(CtrlAviary):  # type: ignore[misc]
    def __init__(
        self,
        *args,
        **kwargs,
        # drone_model: DroneModel = DroneModel.CF2X,
        # num_drones: int = 1,
        # neighbourhood_radius: float = np.inf,
        # initial_xyzs=None,
        # initial_rpys=None,
        # physics: Physics = Physics.PYB,
        # pyb_freq: int = 240,
        # ctrl_freq: int = 240,
        # gui=False,
        # record=False,
        # obstacles=False,
        # user_debug_gui=True,
        # vision_attributes=False,
        # output_folder="results",
    ):

        # WARNING: THE ctrl_freq should match the discretization of the MPC Controller (the dt of the controller in Hz)
        super().__init__(*args, **kwargs)
