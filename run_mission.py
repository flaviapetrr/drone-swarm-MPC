import time
import argparse

import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from mpc.drone_model import get_params, obs_to_state
from mpc.mixer import u_to_rpm
from mpc.controller import MPC
import config as cfg


def run(gui=True, plot=True, duration_sec=cfg.DURATION_SEC):

    #### Create the environment ################################
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=np.array([cfg.INIT_POS]),
                     initial_rpys=np.zeros((1, 3)),
                     physics=Physics.PYB,
                     pyb_freq=cfg.SIM_FREQ_HZ,
                     ctrl_freq=cfg.CTRL_FREQ_HZ,
                     gui=gui,
                     user_debug_gui=False)

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=cfg.CTRL_FREQ_HZ,
                    num_drones=1,
                    output_folder="results",
                    colab=False)

    #### MPC controller ########################################
    P = get_params(env)
    mpc = MPC(P, dt=cfg.DT_MPC, N=cfg.N_MPC, max_tilt_deg=cfg.MAX_TILT_DEG, **cfg.MPC_WEIGHTS)

    #### Run the simulation ####################################
    target = cfg.HOVER_POS    # current MPC target
    phase = 1                 # 1 = hovering, 2 = going to the reference
    settled_steps = 0         # consecutive steps the drone has been steady
    action = np.full((1, 4), P["hover_rpm"])
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        x_now = obs_to_state(obs[0])

        #### Phase logic: hover first, then go to the reference ####
        if phase == 1:
            arrived = (np.linalg.norm(x_now[0:3] - cfg.HOVER_POS) < cfg.SETTLE_TOL
                       and np.linalg.norm(x_now[3:6]) < cfg.SETTLE_VEL)
            settled_steps = settled_steps + 1 if arrived else 0
            if settled_steps >= cfg.SETTLE_SEC * env.CTRL_FREQ:
                phase = 2
                target = cfg.REF_POS
                print(f"t = {i / env.CTRL_FREQ:.2f} s: hover reached, going to the reference position")

        #### Compute the control ###################################
        u, plan = mpc.solve(x_now, target)
        action[0, :] = u_to_rpm(u, P)

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i / env.CTRL_FREQ,
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
    logger.save_as_csv("mpc")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-drone full-state MPC: hover, then go to a reference position")
    parser.add_argument('--gui', default=True, type=str2bool, help='Show the PyBullet GUI (default: True)')
    parser.add_argument('--plot', default=True, type=str2bool, help='Plot the results at the end (default: True)')
    parser.add_argument('--duration_sec', default=cfg.DURATION_SEC, type=int, help='Length of the run in seconds')
    args = parser.parse_args()
    run(**vars(args))