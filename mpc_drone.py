import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cvxpy as cp

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.control.BaseControl import BaseControl
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
from mpc.drone_model import get_params, obs_to_state, continuous_model, input_scale, discretize
from mpc.mixer import allocation_matrix, u_to_rpm, motor_limits

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

#MPC Settings

DT_MPC = 0.05   # [s]
N_MPC = 40
MAX_TILT_DEG = 15.0 # linear model only valid near hover

# ---- Mission
HOVER_POS = np.array([0.0, 0.0, 1.0])   # [m] phase 1: climb to here and hover
REF_POS = np.array([1.0, 0.5, 1.0])   # [m] phase 2: go here after hovering
SETTLE_TOL = 0.05                     # [m] how close in meters to switch to phase 2
SETTLE_VEL = 0.1                      # [m/s] how slowly to be considered settled
SETTLE_SEC = 1.0                      # [s] lenght of time to be considered settled


class MPC:
    def __init__(self, P, dt=DT_MPC, N=N_MPC, w_pos=10., w_vel=1., w_att=1., w_rate=0.1, w_u=1., w_eps=1e3):
        self.N = N
        A, B = continuous_model(P)
        self.u_scale = input_scale(P)
        Ad, Bd = discretize(A, B, dt, self.u_scale)
        F_map, f_hover, f_max = motor_limits(P, self.u_scale)
        xi_max = np.sin(np.radians(MAX_TILT_DEG) / 2)

        self.x = cp.Variable((12, N + 1))
        self.u = cp.Variable((4, N))
        eps = cp.Variable((2, N), nonneg=True)
        self.x0 = cp.Parameter(12)
        self.target = cp.Parameter(3)

        cost = 0.
        constraints = [self.x[:, 0] == self.x0]
        for k in range(N):
            constraints += [self.x[:, k + 1] == Ad @ self.x[:, k] + Bd @ self.u[:, k]]
            f = f_hover + F_map @ self.u[:, k]
            constraints += [f >= 0, f <= f_max]
            constraints += [self.x[6:8, k + 1] <= xi_max + eps[:, k],
                            self.x[6:8, k + 1] >= -xi_max - eps[:, k]]
            cost += (w_pos * cp.sum_squares(self.x[0:3, k] - self.target)
                     + w_vel * cp.sum_squares(self.x[3:6, k])
                     + w_att * cp.sum_squares(self.x[6:9, k])
                     + w_rate * cp.sum_squares(self.x[9:12, k])
                     + w_u * cp.sum_squares(self.u[:, k])
                     + w_eps * cp.sum_squares(eps[:, k]))
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x0, target):
        self.x0.value = x0
        self.target.value = target
        self.problem.solve(solver=cp.OSQP, warm_start=True)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            print("[MPC] solve failed, falling back to hover:", self.problem.status)
            return np.zeros(4), None
        return self.u.value[:, 0] * self.u_scale, self.x.value[0:3, :]

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    #### Initialize the simulation #############################

    INIT_XYZS = np.array([[0.0, 0.0, 0.1]]) #one drone starting at origin, 10cm above groung (1 row x drone)
    INIT_RPYS = np.zeros((1, 3)) #zero roll pitch yaw

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### MPC controller ########################################
    P = get_params(env)
    mpc = MPC(P)

    #### Run the simulation ####################################
    target = HOVER_POS #current MPC target
    phase = 1 #1 for hovering, 2 for going to ref
    settled_steps = 0 #how many consecutive steps drone has been steady
    action = np.full((1, 4), P["hover_rpm"])
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute the control ###################################
        x_now = obs_to_state(obs[0])

        #### Phase logic: hover first, then go to the reference ####
        if phase == 1:
            arrived = (np.linalg.norm(x_now[0:3] - HOVER_POS) < SETTLE_TOL
                       and np.linalg.norm(x_now[3:6]) < SETTLE_VEL)
            settled_steps = settled_steps + 1 if arrived else 0
            if settled_steps >= SETTLE_SEC * env.CTRL_FREQ:
                phase = 2
                target = REF_POS
                print(f"t = {i / env.CTRL_FREQ:.2f} s: hover reached, going to the reference position")

        u, plan = mpc.solve(x_now, target)
        action[0, :] = u_to_rpm(u, P)

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=obs[0],
                   control=np.hstack([target, np.zeros(9)]))

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))