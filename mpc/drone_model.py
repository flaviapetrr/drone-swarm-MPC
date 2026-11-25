"""Drone parameters, state conversion and the linear model """
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

def get_params(env):
    """Physical parameters read from the simulator, so the MPC model always matches it."""
    return dict(m=env.M, g=env.G, J=env.J,
                arm=env.L, kf=env.KF, km=env.KM,
                hover_rpm=env.HOVER_RPM, max_rpm=env.MAX_RPM)

def obs_to_state(obs_row):
    """report's 12-state [x y z, vx vy vz, xi1 xi2 xi3, p q r]."""
    pos, quat = obs_row[0:3], obs_row[3:7]            # PyBullet quaternion order is (x, y, z, w)
    vel, w_world = obs_row[10:13], obs_row[13:16]
    xi_v = quat[0:3] if quat[3] >= 0 else -quat[0:3]  # q and -q are the same rotation; the report assumes xi_0 > 0
    w_body = Rotation.from_quat(quat).as_matrix().T @ w_world   # the report uses body-frame rates
    return np.hstack([pos, vel, xi_v, w_body])

def continuous_model(P):
    """A and B of the report's linear model around hover (eq. 23-24). Input: [du1, tau_x, tau_y, tau_z]."""
    m, g = P["m"], P["g"]
    I3, Z = np.eye(3), np.zeros((3, 3))
    G = np.array([[0.,  2*g, 0.],
                  [-2*g, 0., 0.],
                  [0.,   0., 0.]])
    A = np.block([[Z, I3, Z, Z],
                  [Z, Z,  G, Z],
                  [Z, Z,  Z, 0.5*I3],
                  [Z, Z,  Z, Z]])
    B = np.zeros((12, 4))
    B[5, 0] = 1 / m                       # thrust -> z acceleration
    B[9:12, 1:4] = np.linalg.inv(P["J"])  # torques -> angular accelerations
    return A, B


def input_scale(P):
    """Typical size of [du1, tau_x, tau_y, tau_z]. The MPC will work with u / scale (numbers around 1)."""
    return np.array([P["m"] * P["g"], 3e-3, 3e-3, 1e-3])


def discretize(A, B, dt, u_scale):
    """Exact zero-order-hold discretisation of x' = A x + B (u_scale * u_tilde)."""
    n, m = B.shape
    E = expm(np.block([[A, B @ np.diag(u_scale)], [np.zeros((m, n + m))]]) * dt)
    return E[:n, :n], E[:n, n:]          # Ad, Bd (acting on the scaled input)
