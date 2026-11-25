import time
import sys
import os
import numpy as np

from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.drone_model import OurModel
from src.mpc_controller import MPCController


import pybullet as p


def draw_prediction_points(client_id, positions, debug_item_ids):
    for item_id in debug_item_ids:
        p.removeUserDebugItem(item_id, physicsClientId=client_id)
    debug_item_ids.clear()

    points = positions.T.tolist()
    if not points:
        return

    item_id = p.addUserDebugPoints(
        pointPositions=points,
        pointColorsRGB=[[1.0, 0, 0]] * len(points),
        pointSize=2,
        lifeTime=0,
        physicsClientId=client_id,
    )
    debug_item_ids.append(item_id)


DRONE = DroneModel.CF2X
NUM_DRONES = 1
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 3
TARGET_POS = np.array([0.0, 0.0, 2.0])

initial_xyzs = np.array([[0.0, 0.0, 0.1]])
initial_rpys = np.zeros((NUM_DRONES, 3))

env = CtrlAviary(
    drone_model=DRONE,
    num_drones=NUM_DRONES,
    initial_xyzs=initial_xyzs,
    initial_rpys=initial_rpys,
    physics=Physics.PYB,
    pyb_freq=SIMULATION_FREQ_HZ,
    ctrl_freq=CONTROL_FREQ_HZ,
    gui=True,
    record=True,
    output_folder="results",
    obstacles=False,
    user_debug_gui=False,
)

controller = MPCController(
    drone_model=DRONE,
    horizon=10,
    dt=env.CTRL_TIMESTEP,
    min_rpm=0.0,
    max_rpm=env.MAX_RPM,
    hover_rpm=env.HOVER_RPM,
)


logger = Logger(
    logging_freq_hz=env.CTRL_FREQ,
    num_drones=NUM_DRONES,
    output_folder="results",
)

try:
    obs, info = env.reset()

    client_id = env.getPyBulletClient()
    time.sleep(2)
    prediction_line_ids = []

    # Controller returns four RPMs for this one drone.
    action = np.zeros((NUM_DRONES, 4), dtype=np.float32)
    action[0] = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=TARGET_POS,
    )

    start = time.time()

    for i in range(DURATION_SEC * env.CTRL_FREQ):
        obs, reward, terminated, truncated, info = env.step(action)

        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
        )

        action[0] = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS,
        )

        draw_prediction_points(
            client_id,
            controller.predicted_positions,
            prediction_line_ids,
        )

        env.render()

        if env.GUI:
            sync(i, start, env.CTRL_TIMESTEP)

        if terminated or truncated:
            break
finally:
    env.close()

    logger.save()
    logger.save_as_csv("mpc")
