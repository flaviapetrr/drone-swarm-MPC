# Basic imports we need

import time
import sys
import os
from typing import Sequence
import numpy as np
import cvxpy as cp

# Imports we need for gym - IGNORE
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

# Setting the directory, lowk just copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We import our drone model and the MPC controller from src/
from src.drone_model import OurModel
from src.mpc_controller import MPCController
import pybullet as p

# Here we import the rule set we create, its important to keep it in a seperate file because it
# allows us to interchange the different rule sets without copying entire example.py
import examples.rules.hover as hover


# IGNORE - Draws points for the future predicitions from MPC
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


# IMPORTANT - We define the variables of the experiment
DRONE = DroneModel.CF2X
NUM_DRONES = 1
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 3
TARGET_POS = np.array([0.0, 0.0, 2.0])
HORIZON = 10

# Initial position of the drone
initial_xyzs = np.array([[0.0, 0.0, 0.1]])

# I have no idea what this is, we dont use it, but we need it
initial_rpys = np.zeros((NUM_DRONES, 3))

# The drone model, for now its a really simple model
env = OurModel(
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

# The MPCConctoller model, simple stuff really, just important that we use the env.CTRL_TIMESTEP
controller = MPCController(
    drone_model=DRONE,
    horizon=HORIZON,
    dt=env.CTRL_TIMESTEP,
    min_rpm=0.0,
    max_rpm=env.MAX_RPM,
    hover_rpm=env.HOVER_RPM,
)

# IMPORTANT - the apply function basically says when to apply the constraints and costs in the horizon (0...k)
# We can change the arguments of the function to add some extra stuff that is needed
hover.apply(controller, target_pos=TARGET_POS, horizon=HORIZON)


# A logger thing, just saves the data for us to analyse after
logger = Logger(
    logging_freq_hz=env.CTRL_FREQ,
    num_drones=NUM_DRONES,
    output_folder="results",
)

try:
    obs, info = env.reset()

    # Setup the GUI
    client_id = env.getPyBulletClient()
    time.sleep(2)
    prediction_line_ids = []

    # Get the first action, honestly idk why we do this yet
    action = np.zeros((NUM_DRONES, 4), dtype=np.float32)
    action[0] = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=TARGET_POS,
    )

    # Start recording the time it takes
    start = time.time()

    for i in range(DURATION_SEC * env.CTRL_FREQ):
        # Basically the step function that takes an action -> important is the obs, which is basically the state
        obs, reward, terminated, truncated, info = env.step(action)

        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
        )

        # This function call just computes the next action
        action[0] = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS,
        )

        # draw the predicted positions
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
