"""Control allocation: [thrust, torques] <-> motor forces and RPMs, and the motor limits."""
import numpy as np

def allocation_matrix(P):
    """Gamma: [T, tau_x, tau_y, tau_z] = Gamma @ [f0, f1, f2, f3]  (f_i = per-motor thrust in N)."""
    d = P["arm"] / np.sqrt(2)      # motors sit at (+-d, +-d)
    c = P["km"] / P["kf"]          # yaw torque per unit of thrust
    return np.array([[ 1,  1,  1,  1],
                     [-d, -d,  d,  d],
                     [-d,  d,  d, -d],
                     [-c,  c, -c,  c]])


def u_to_rpm(u, P):
    """u = [du1, tau_x, tau_y, tau_z] in physical units (N, N*m) -> 4 motor RPMs."""
    wrench = np.array([P["m"] * P["g"] + u[0], u[1], u[2], u[3]])
    f = np.linalg.solve(allocation_matrix(P), wrench)
    f = np.clip(f, 0.0, P["kf"] * P["max_rpm"] ** 2)      # safety net: motors cannot pull or exceed max RPM
    return np.sqrt(f / P["kf"])


def motor_limits(P, u_scale):
    """Motor limits as linear functions of the scaled input: f = f_hover + F_map @ u_tilde, 0 <= f <= f_max."""
    F_map = np.linalg.solve(allocation_matrix(P), np.diag(u_scale))
    f_hover = np.full(4, P["m"] * P["g"] / 4)
    f_max = np.full(4, P["kf"] * P["max_rpm"] ** 2)
    return F_map, f_hover, f_max