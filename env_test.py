import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    initial_xyzs=np.array([[0, 0, 1.0]]),
    physics=Physics.PYB,
    pyb_freq=960, ctrl_freq=48,
    gui=False,
)

obs, info = env.reset()
p.changeDynamics(env.DRONE_IDS[0], -1, linearDamping=0, angularDamping=0, physicsClientId=env.CLIENT)
# print("obs shape:", obs.shape)
# print("action space:", env.action_space) # shape [num drones, 4]
# print("obs space:", env.observation_space)
# print("mass, kf, km, hover rpm:", env.M, env.KF, env.KM, env.HOVER_RPM)
# print("ctrl dt, pyb dt:", env.CTRL_TIMESTEP, env.PYB_TIMESTEP)
print ("env.M, env.GRAVITY, env.KF: ", env.M, env.GRAVITY, env.KF)

T = 2 # sec
z = [] # to store the z position of the drone
z_action = env.HOVER_RPM + (env.HOVER_RPM * 0.01) # increase the hover rpm by 1%
for i in range(int(T/env.CTRL_TIMESTEP)):
    action =  np.full((1,4), z_action)
    obs, reward, done, truncated, terminated = env.step(action)
    z.append(float(obs[0, 2])) # store the z position of the drone

env.close()
print ("z[-1] - z[0]: ", z[-1] - z[0])
HOVER_RPM_check = np.sqrt(env.GRAVITY / (4 * env.KF)) # env.GRAVITY it is already env.M * gravity
print("HOVER_RPM_check: ", np.isclose(HOVER_RPM_check, env.HOVER_RPM))
plt.plot(np.arange(len(z))*env.CTRL_TIMESTEP, z)
plt.show()