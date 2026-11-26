"""Try: a single drone takes off from the ground and then follows a reference.

Two reference trajectories (flag TRAJECTORY or --traj):
  "linear"   : vertical climb to TARGET_Z (minimum-jerk profile), then hover.
  "circular" : same climb, then a vertical circle ("loop") whose lowest point
               is the hover point. The drone stays almost level: the model is
               linearized around hover, so a real flip is NOT possible.

Controller: linear MPC (linearized around hover), solved with cvxpy/OSQP.
The problem is built ONCE with cp.Parameter and then only updated at each
step (much faster than rebuilding it every time).

Model = the one in the preliminary report (Sec. V, eq. 21-24):
State  x  (12): [x, y, z, xd, yd, zd, xi1, xi2, xi3, p, q, r]
                 position, velocity (world), vector part of the quaternion,
                 angular velocity in the body frame
Input  du (4) : [du1, u2, u3, u4]
                 du1      = total thrust - m*g             [N]
                 u2,u3,u4 = body-frame torques             [N*m]
x_dot = A x + B du,  B = [0; e3/m; 0; J^-1].

The reference contains position, velocity AND the feed-forward terms obtained
by inverting the linear model (eq. 11-13):
    xi2_ref = xdd_ref / (2g),  xi1_ref = -ydd_ref / (2g),  du1_ref = m * zdd_ref
so the MPC does not have to "lag behind" to generate the needed acceleration.

Usage:
    python tries/takeoff_hover.py                      # GUI, TRAJECTORY below
    python tries/takeoff_hover.py --traj circular      # override the flag
    python tries/takeoff_hover.py --no-gui             # headless, metrics only
    python tries/takeoff_hover.py --record             # also save an MP4 (needs ffmpeg)
"""

import argparse
import os
import shutil
import time
from datetime import datetime

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.linalg import expm

from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

# ===================================================================
#  TUNABLE PARAMETERS (everything you may want to change is here)
# ===================================================================

TRAJECTORY = "linear"                   # "linear" | "circular"

# ----------------------------------------------------------- simulation
DRONE = DroneModel.CF2X
PHYSICS = Physics.PYB                   # PYB, DYN, PYB_GND, PYB_DRAG, ...
SIM_FREQ_HZ = 240
CTRL_FREQ_HZ = 48                       # MPC rate; dt = 1 / CTRL_FREQ_HZ
DURATION_SEC = 8.0                      # "linear" only; "circular" is computed

# ------------------------------------------------------ scenario: climb
START_POS = np.array([0.0, 0.0, 0.02])  # resting on the ground
TARGET_XY = START_POS[0:2]              # horizontal position to hold [m]
TARGET_Z = 2.0                          # hover altitude [m]
CLIMB_TIME = 1.0                        # climb duration [s]

# --------------------------------------------- scenario: vertical circle
# The lowest point of the circle is (TARGET_XY, TARGET_Z).
CIRCLE_PLANE = "xz"                     # vertical plane: "xz" or "yz"
CIRCLE_RADIUS = 0.5                     # [m]
CIRCLE_PERIOD = 4.0                     # time per lap at full speed [s]
CIRCLE_LAPS = 2                         # number of laps at full speed
CIRCLE_START_DELAY = 1.0                # hover at the bottom before starting [s]
CIRCLE_RAMP_TIME = 2.0                  # angular speed ramps 0 -> 2*pi/PERIOD [s]
CIRCLE_END_HOVER = 1.0                  # simulated time after the last lap [s]

# --------------------------------------------------------------------- MPC
HORIZON = 20                            # 20 * (1/48 s) ~ 0.42 s prediction
MAX_VZ = 1.0                            # max |v_z| [m/s]
MAX_TILT_DEG = 20.0                     # max tilt angle
MIN_MOTOR_FORCE = 0.0                   # per-motor thrust lower bound [N]
SOLVER = cp.OSQP
WARM_START = True

#              pos            vel          xi_v         omega
Q = np.diag([20, 20, 60,   2, 2, 6,   10, 10, 5,   0.1, 0.1, 0.1])
# Input weights expressed as penalties on the equivalent accelerations
# (du1/m [m/s^2], u_i/I_ii [rad/s^2]); TrackingMPC converts them into weights
# on [N, N*m]: R = diag(R_ACC / [m, Ixx, Iyy, Izz]^2).
R_ACC = np.array([0.5, 1e-4, 1e-4, 1e-4])
TERMINAL_WEIGHT = 10.0                  # terminal cost = TERMINAL_WEIGHT * Q

# -------------------------------------------------------------- metrics
SETTLING_BAND = 0.02                    # settling band, fraction of the climb
HOVER_EVAL_DELAY = 1.0                  # hover error measured from CLIMB_TIME + this [s]

# ----------------------------------------------------------------- output
SHOW_PREDICTION = True                  # draw predicted trajectory in the GUI (red)
SHOW_REFERENCE = True                   # draw the reference path in the GUI (green)
PLOT_FILE = "takeoff_hover_{traj}.png"  # saved in results/
RECORD_VIDEO = False                    # MP4 of the GUI window (needs GUI + ffmpeg in PATH)
VIDEO_FILE = "takeoff_hover_{traj}_{stamp}.mp4"  # saved in results/ (NO spaces in the name)

# ----------------------------------------------------------------- camera (GUI)
CAMERA_DISTANCE = 4.0                   # [m]
CAMERA_YAW = 0.0                        # [deg] 0 = looking along +y (sees the x-z plane)
CAMERA_PITCH = -15.0                    # [deg]
CAMERA_TARGET = None                    # [x, y, z]; None = middle of the trajectory


# ===================================================================
#  REFERENCE TRAJECTORIES
#  Each returns (pos, vel, acc), arrays of shape (3, len(t)).
# ===================================================================

def climb_reference(t, p0):
    """Vertical quintic (minimum-jerk) climb from p0 to (TARGET_XY, TARGET_Z) in
    CLIMB_TIME, then constant. Zero velocity and acceleration at both ends."""
    t = np.asarray(t, dtype=float)
    T = CLIMB_TIME
    dz = TARGET_Z - p0[2]
    s = np.clip(t / T, 0.0, 1.0)
    inside = (t >= 0) & (t <= T)
    pos = np.zeros((3, t.size))
    vel = np.zeros((3, t.size))
    acc = np.zeros((3, t.size))
    pos[0], pos[1] = TARGET_XY[0], TARGET_XY[1]
    pos[2] = p0[2] + dz * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel[2] = np.where(inside, dz / T * (30 * s**2 - 60 * s**3 + 30 * s**4), 0.0)
    acc[2] = np.where(inside, dz / T**2 * (60 * s - 180 * s**2 + 120 * s**3), 0.0)
    return pos, vel, acc


def circle_start_time():
    return CLIMB_TIME + CIRCLE_START_DELAY


def circle_end_time():
    # the ramp covers half the angle of a full-speed phase of the same length
    return circle_start_time() + CIRCLE_RAMP_TIME + CIRCLE_LAPS * CIRCLE_PERIOD


def circle_axes():
    """Unit vectors (horizontal, vertical) spanning the circle plane."""
    e_h = np.array([1.0, 0.0, 0.0]) if CIRCLE_PLANE == "xz" else np.array([0.0, 1.0, 0.0])
    return e_h, np.array([0.0, 0.0, 1.0])


def circular_reference(t, p0):
    """Climb, hover at the bottom, then a vertical circle with a smooth
    start: the angular speed grows linearly from 0 to w in CIRCLE_RAMP_TIME."""
    t = np.asarray(t, dtype=float)
    pos, vel, acc = climb_reference(t, p0)

    w = 2 * np.pi / CIRCLE_PERIOD
    Tr = CIRCLE_RAMP_TIME
    tau = t - circle_start_time()
    on = (tau >= 0) & (t <= circle_end_time())
    tau = np.clip(tau, 0.0, None)

    # angle theta(tau), its rate and acceleration (theta = -pi/2 is the bottom)
    ramp = tau < Tr
    th = -np.pi / 2 + np.where(ramp, w * tau**2 / (2 * Tr), w * (tau - Tr / 2))
    thd = np.where(ramp, w * tau / Tr, w)
    thdd = np.where(ramp, w / Tr, 0.0)

    e_h, e_v = circle_axes()
    bottom = np.array([TARGET_XY[0], TARGET_XY[1], TARGET_Z])
    center = bottom + CIRCLE_RADIUS * e_v
    c, s = np.cos(th), np.sin(th)
    R = CIRCLE_RADIUS
    p_c = center[:, None] + R * (np.outer(e_h, c) + np.outer(e_v, s))
    v_c = R * thd * (np.outer(e_h, -s) + np.outer(e_v, c))
    a_c = (R * thdd * (np.outer(e_h, -s) + np.outer(e_v, c))
           - R * thd**2 * (np.outer(e_h, c) + np.outer(e_v, s)))

    # after the last lap: hold the point where the circle ended
    after = t > circle_end_time()
    if np.any(after):
        p_end, _, _ = circular_reference(np.array([circle_end_time()]), p0)
        pos[:, after] = p_end
        vel[:, after] = 0.0
        acc[:, after] = 0.0
    pos[:, on], vel[:, on], acc[:, on] = p_c[:, on], v_c[:, on], a_c[:, on]
    return pos, vel, acc


REFERENCES = {"linear": climb_reference, "circular": circular_reference}


def check_circle_feasibility(g, max_acc_up):
    """Warn if the circle asks for more than the linear model / motors allow."""
    w = 2 * np.pi / CIRCLE_PERIOD
    R = CIRCLE_RADIUS
    a_centripetal = R * w**2
    a_tangential = R * w / CIRCLE_RAMP_TIME
    # horizontal acc. from tilt: xdd = 2 g xi2, |xi2| <= sin(MAX_TILT/2)
    a_h_max = 2 * g * np.sin(np.deg2rad(MAX_TILT_DEG) / 2)
    print(f"circle: v = {R * w:.2f} m/s, centripetal acc = {a_centripetal:.2f} m/s^2")
    if np.hypot(a_centripetal, a_tangential) > 0.8 * a_h_max:
        print(f"  WARNING: horizontal acc. close to/above the tilt limit "
              f"({a_h_max:.2f} m/s^2 at {MAX_TILT_DEG} deg)")
    if a_centripetal > g:
        print("  WARNING: at the top the drone would need negative thrust "
              "(a real flip): not possible with this model")
    if a_centripetal > max_acc_up:
        print(f"  WARNING: at the bottom the motors can give only "
              f"{max_acc_up:.2f} m/s^2 upwards")
    if R * w > MAX_VZ:
        print(f"  WARNING: peak |v_z| = {R * w:.2f} m/s > MAX_VZ = {MAX_VZ} m/s")


# ===================================================================
#  MPC
# ===================================================================

class TrackingMPC:
    def __init__(self, env: CtrlAviary, horizon: int, dt: float):
        self.N = horizon
        self.dt = dt
        m, g = env.M, env.G
        self.m, self.g = m, g
        Ixx, Iyy, Izz = np.diag(env.J)
        self.KF = env.KF
        self.MAX_RPM = env.MAX_RPM

        # --- continuous-time model linearized around hover (report, eq. 23-24)
        z3, I3 = np.zeros((3, 3)), np.eye(3)
        # xi1 ~ roll/2, xi2 ~ pitch/2
        # roll>0 -> acceleration along -y ; pitch>0 -> acceleration along +x
        G = np.array([[0, 2 * g, 0], [-2 * g, 0, 0], [0, 0, 0]])
        A = np.block([[z3, I3, z3, z3],
                      [z3, z3, G, z3],
                      [z3, z3, z3, 0.5 * I3],
                      [z3, z3, z3, z3]])
        B = np.zeros((12, 4))
        B[5, 0] = 1.0 / m                          # du1 -> zdd   (e3/m)
        B[9:12, 1:4] = np.diag([1 / Ixx, 1 / Iyy, 1 / Izz])  # torques -> omega_dot (J^-1)

        # --- exact discretization (zero-order hold)
        M = np.zeros((16, 16))
        M[:12, :12], M[:12, 12:] = A, B
        Md = expm(M * dt)
        self.Ad, self.Bd = Md[:12, :12], Md[:12, 12:]

        # --- allocation: [du1, u2, u3, u4] -> forces of the 4 motors
        # same convention as BaseAviary._dynamics / DSLPIDControl (CF2X)
        l = env.L / np.sqrt(2)
        k = env.KM / env.KF
        wrench_from_f = np.array([[1, 1, 1, 1],
                                  [-l, -l, l, l],
                                  [-l, l, l, -l],
                                  [-k, k, -k, k]])
        self.F_FROM_U = np.linalg.inv(wrench_from_f)
        self.F_HOVER = m * g / 4.0
        f_max = self.KF * self.MAX_RPM**2
        self.max_acc_up = 4 * f_max / m - g

        # --- optimization problem (built only once)
        N = self.N
        self.x = cp.Variable((12, N + 1))
        self.u = cp.Variable((4, N))
        self.x0 = cp.Parameter(12)
        self.xref = cp.Parameter((12, N + 1))
        self.uref = cp.Parameter((4, N))       # feed-forward input

        R = np.diag(R_ACC / np.array([m, Ixx, Iyy, Izz]) ** 2)
        Qs, Rs = np.sqrt(Q), np.sqrt(R)
        q_tilt = np.sin(np.deg2rad(MAX_TILT_DEG) / 2)
        cost = 0
        cons = [self.x[:, 0] == self.x0]
        for j in range(N):
            cost += cp.sum_squares(Qs @ (self.x[:, j] - self.xref[:, j]))
            cost += cp.sum_squares(Rs @ (self.u[:, j] - self.uref[:, j]))
            f = self.F_HOVER + self.F_FROM_U @ self.u[:, j]
            cons += [
                self.x[:, j + 1] == self.Ad @ self.x[:, j] + self.Bd @ self.u[:, j],
                f >= MIN_MOTOR_FORCE,
                f <= f_max,
                cp.abs(self.x[5, j + 1]) <= MAX_VZ,
                cp.abs(self.x[6:8, j + 1]) <= q_tilt,
            ]
        cost += TERMINAL_WEIGHT * cp.sum_squares(Qs @ (self.x[:, N] - self.xref[:, N]))
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        assert self.prob.is_dpp()

    def build_reference(self, pos, vel, acc):
        """(pos, vel, acc) over the horizon -> (xref, uref) by inverting the
        linear model: xdd = 2g xi2, ydd = -2g xi1, zdd = du1 / m."""
        xref = np.zeros((12, self.N + 1))
        xref[0:3] = pos
        xref[3:6] = vel
        xref[6] = -acc[1] / (2 * self.g)
        xref[7] = acc[0] / (2 * self.g)
        uref = np.zeros((4, self.N))
        uref[0] = self.m * acc[2, :self.N]
        return xref, uref

    @staticmethod
    def state_from_obs(obs):
        """BaseAviary obs: pos[0:3] quat[3:7](x,y,z,w) rpy[7:10] vel[10:13] ang_v[13:16]."""
        pos, quat, vel, ang_v_world = obs[0:3], obs[3:7], obs[10:13], obs[13:16]
        if quat[3] < 0:  # same orientation, sign chosen so that w>=0
            quat = -quat
        Rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega_body = Rot.T @ ang_v_world  # pybullet returns omega in the world frame
        return np.concatenate([pos, vel, quat[0:3], omega_body])

    def compute(self, obs, xref, uref):
        self.x0.value = self.state_from_obs(obs)
        self.xref.value = xref
        self.uref.value = uref
        self.prob.solve(solver=SOLVER, warm_start=WARM_START)
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC solve failed: {self.prob.status}")
        u0 = self.u[:, 0].value
        f = np.clip(self.F_HOVER + self.F_FROM_U @ u0, MIN_MOTOR_FORCE, self.KF * self.MAX_RPM**2)
        return np.sqrt(f / self.KF), self.x.value


# ===================================================================
#  MAIN
# ===================================================================

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def main(gui: bool, traj: str, record: bool):
    reference = REFERENCES[traj]
    if record and not gui:
        print("WARNING: video recording needs the GUI; --no-gui given, no video.")
        record = False
    if record and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found in PATH, no video will be saved "
              "(install it with: winget install ffmpeg, then reopen the terminal).")
        record = False
    duration = DURATION_SEC if traj == "linear" else circle_end_time() + CIRCLE_END_HOVER

    env = CtrlAviary(drone_model=DRONE, num_drones=1,
                     initial_xyzs=START_POS.reshape(1, 3),
                     initial_rpys=np.zeros((1, 3)),
                     physics=PHYSICS, pyb_freq=SIM_FREQ_HZ, ctrl_freq=CTRL_FREQ_HZ,
                     gui=gui, record=False, obstacles=False, user_debug_gui=False)
    dt = env.CTRL_TIMESTEP
    mpc = TrackingMPC(env, HORIZON, dt)
    vz_peak = 1.875 * abs(TARGET_Z - START_POS[2]) / CLIMB_TIME  # max of the quintic
    if vz_peak > MAX_VZ:
        print(f"WARNING: the climb asks for |v_z| = {vz_peak:.2f} m/s > MAX_VZ = {MAX_VZ} m/s; "
              f"the drone will lag behind (use CLIMB_TIME >= "
              f"{1.875 * abs(TARGET_Z - START_POS[2]) / MAX_VZ:.1f} s)")
    if traj == "circular":
        check_circle_feasibility(env.G, mpc.max_acc_up)

    n_steps = int(duration * CTRL_FREQ_HZ)
    log_t, log_pos, log_ref, log_tilt, log_rpm, log_solve = [], [], [], [], [], []

    obs, _ = env.reset()
    p0 = obs[0][0:3].copy()
    action = np.zeros((1, 4))
    pred_id = None

    if gui and SHOW_REFERENCE:
        path, _, _ = reference(np.linspace(0, duration, 400), p0)
        for a, b in zip(path.T[:-1], path.T[1:]):
            p.addUserDebugLine(a.tolist(), b.tolist(), [0, 0.8, 0], lineWidth=2,
                               physicsClientId=env.CLIENT)

    if gui:
        if CAMERA_TARGET is None:
            path, _, _ = reference(np.linspace(0, duration, 400), p0)
            target = (path.min(axis=1) + path.max(axis=1)) / 2
        else:
            target = CAMERA_TARGET
        p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW,
                                     cameraPitch=CAMERA_PITCH,
                                     cameraTargetPosition=list(target),
                                     physicsClientId=env.CLIENT)

    video_id = None
    if record:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        video_name = VIDEO_FILE.format(traj=traj, stamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
        # pybullet passes the file name to ffmpeg WITHOUT quotes, so a path with
        # spaces breaks it. ffmpeg is spawned here and inherits the current
        # directory: start it from RESULTS_DIR and give only the bare file name.
        prev_cwd = os.getcwd()
        os.chdir(RESULTS_DIR)
        try:
            video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_name,
                                           physicsClientId=env.CLIENT)
        finally:
            os.chdir(prev_cwd)
        print(f"recording video     : {os.path.join(RESULTS_DIR, video_name)}")

    start = time.time()
    try:
        for i in range(n_steps):
            t = i * dt
            t_h = t + dt * np.arange(HORIZON + 1)
            pos_r, vel_r, acc_r = reference(t_h, p0)
            xref, uref = mpc.build_reference(pos_r, vel_r, acc_r)

            tic = time.perf_counter()
            action[0], x_pred = mpc.compute(obs[0], xref, uref)
            log_solve.append(time.perf_counter() - tic)

            roll, pitch = obs[0][7], obs[0][8]
            log_t.append(t)
            log_pos.append(obs[0][0:3].copy())
            log_ref.append(pos_r[:, 0])
            log_tilt.append(np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1))))
            log_rpm.append(action[0].copy())

            if gui and SHOW_PREDICTION:
                if pred_id is not None:
                    p.removeUserDebugItem(pred_id, physicsClientId=env.CLIENT)
                pred_id = p.addUserDebugPoints(x_pred[0:3].T.tolist(),
                                               [[1, 0, 0]] * (HORIZON + 1), pointSize=3,
                                               physicsClientId=env.CLIENT)

            obs, _, terminated, truncated, _ = env.step(action)
            if gui:
                sync(i, start, dt)
            if terminated or truncated:
                break
    finally:
        if video_id is not None:
            p.stopStateLogging(video_id, physicsClientId=env.CLIENT)
        env.close()

    # ------------------------------------------------------------- metrics
    t = np.array(log_t)
    pos, ref = np.array(log_pos), np.array(log_ref)
    tilt, rpm = np.array(log_tilt), np.array(log_rpm)
    err = np.linalg.norm(pos - ref, axis=1)

    print(f"trajectory          : {traj}")
    if traj == "linear":
        z, zref = pos[:, 2], ref[:, 2]
        ez = z - TARGET_Z
        band = SETTLING_BAND * (TARGET_Z - p0[2])
        outside = np.where(np.abs(ez) > band)[0]
        settle = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else np.nan
        hover = t >= CLIMB_TIME + HOVER_EVAL_DELAY
        xy_drift = np.linalg.norm(pos[:, 0:2] - TARGET_XY, axis=1)
        print(f"quota finale        : {z[-1]:.4f} m (target {TARGET_Z} m)")
        print(f"overshoot           : {max(0.0, z.max() - TARGET_Z) * 100:.2f} cm")
        print(f"tracking RMS salita : {np.sqrt(np.mean((z - zref)[t < CLIMB_TIME] ** 2)) * 100:.2f} cm")
        print(f"settling ({SETTLING_BAND:.0%})       : {settle:.2f} s")
        print(f"errore RMS in hover : {np.sqrt(np.mean(ez[hover] ** 2)) * 1000:.2f} mm")
        print(f"drift xy max        : {xy_drift.max() * 1000:.2f} mm")
    else:
        on = (t >= circle_start_time()) & (t <= circle_end_time())
        print(f"errore RMS cerchio  : {np.sqrt(np.mean(err[on] ** 2)) * 100:.2f} cm")
        print(f"errore max cerchio  : {err[on].max() * 100:.2f} cm")
        print(f"errore finale       : {err[-1] * 100:.2f} cm")
    print(f"inclinazione max    : {tilt.max():.1f} deg (limite {MAX_TILT_DEG} deg)")
    print(f"tempo solve MPC     : media {np.mean(log_solve) * 1e3:.1f} ms, "
          f"max {np.max(log_solve) * 1e3:.1f} ms (budget {dt * 1e3:.1f} ms)")

    # ---------------------------------------------------------------- plot
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    if traj == "linear":
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(t, ref[:, 2], "--", label="z_ref")
        ax[0].plot(t, pos[:, 2], label="z")
        ax[0].set_ylabel("quota [m]")
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(t, rpm)
        ax[1].set_ylabel("RPM motori")
        ax[1].set_xlabel("t [s]")
        ax[1].grid(True)
    else:
        h = 0 if CIRCLE_PLANE == "xz" else 1
        fig = plt.figure(figsize=(11, 6))
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.plot(ref[:, h], ref[:, 2], "--", label="riferimento")
        ax0.plot(pos[:, h], pos[:, 2], label="drone")
        ax0.set_xlabel(f"{CIRCLE_PLANE[0]} [m]")
        ax0.set_ylabel("z [m]")
        ax0.set_aspect("equal")
        ax0.legend()
        ax0.grid(True)
        ax1 = fig.add_subplot(3, 2, 2)
        ax1.plot(t, err * 100)
        ax1.set_ylabel("errore [cm]")
        ax1.grid(True)
        ax2 = fig.add_subplot(3, 2, 4, sharex=ax1)
        ax2.plot(t, tilt)
        ax2.axhline(MAX_TILT_DEG, color="r", ls=":")
        ax2.set_ylabel("tilt [deg]")
        ax2.grid(True)
        ax3 = fig.add_subplot(3, 2, 6, sharex=ax1)
        ax3.plot(t, rpm)
        ax3.set_ylabel("RPM motori")
        ax3.set_xlabel("t [s]")
        ax3.grid(True)
        for a in (ax1, ax2, ax3):
            a.axvspan(circle_start_time(), circle_end_time(), color="0.9")
    fig.tight_layout()
    path = os.path.join(out_dir, PLOT_FILE.format(traj=traj))
    fig.savefig(path, dpi=120)
    print(f"grafico salvato in  : {os.path.abspath(path)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--traj", choices=list(REFERENCES), default=TRAJECTORY)
    ap.add_argument("--record", action="store_true", help="save an MP4 of the GUI")
    args = ap.parse_args()
    main(gui=not args.no_gui, traj=args.traj, record=RECORD_VIDEO or args.record)
