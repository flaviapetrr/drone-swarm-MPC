from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import numpy as np
import cvxpy as cp


class MPCDynamics:
    def __init__(
        self, Ixx: float, Iyy: float, Izz: float, m, g=9.81, dtype=np.float32
    ) -> None:
        z33 = np.zeros((3, 3), dtype)
        z31 = np.zeros((3, 1), dtype)
        i33 = np.eye(3, dtype=dtype)
        e3 = np.array([[0.0, 0.0, 1.0]]).T

        J_inv = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz]).astype(dtype)
        G = np.array(
            [[0.0, 2.0 * g, 0.0], [-2.0 * g, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype
        )

        self.A = np.block(
            [
                [z33, i33, z33, z33],
                [z33, z33, G, z33],
                [z33, z33, z33, 0.5 * i33],
                [z33, z33, z33, z33],
            ]
        )

        self.B = np.block([[z31, z33], [1 / m * e3, z33], [z31, z33], [z31, J_inv]])

    def next_x(self, x, u):
        return self.A @ x + self.B @ u


class MPCController(BaseControl):
    """
    MPC controller for the group

    This is a cascaded position and attitude controller. The ``target_rpy_rates``
    input is interpreted as roll, pitch, and yaw Euler-angle rates in rad/s and
    is used by the derivative term of the attitude PID. It is not a body-rate
    (p, q, r) or ACRO command interface, and ``cur_ang_vel`` is currently unused.

    """

    def __init__(
        self,
        drone_model: DroneModel,
        horizon,
        g: float = 9.8,
        min_rpm: float = 0.0,
        max_rpm: float = 0.0,
        hover_rpm: float = 0.0,
    ) -> None:
        super().__init__(drone_model=drone_model, g=g)

        assert (
            self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P
        )

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array(
                [
                    [-0.5, -0.5, -1],
                    [-0.5, 0.5, 1],
                    [0.5, 0.5, -1],
                    [0.5, -0.5, 1],
                ]
            )
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array(
                [
                    [0, -1, -1],
                    [+1, 0, 1],
                    [0, 1, -1],
                    [-1, 0, 1],
                ]
            )

        self.MIN_RPM = min_rpm
        self.MAX_RPM = max_rpm
        self.HOVER_RPM = hover_rpm

        # DEFINE THE VARIABLES HERE
        #
        # The state composes of
        # - pos : [3x1] {m}
        # - quaternions : [3x1] {} - 3 independent
        # - vel : [3x1] {m/s}
        # - ang_vel : [3x1] p,q,r
        #
        # The gym API gives us
        # state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
        # self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])

        self.x = cp.Variable((12, horizon + 1))
        # self.x = np.zeros((12, 1), dtype=np.float32)

        # The control_input composes of
        # delta_thrust : 1x1 {N}
        # torques : 3x1 {Nm}

        self.u = cp.Variable((4, horizon))
        # self.u = np.zeros((4, 1), dtype=np.float32)

        # Cost
        self.cost = 0

        # Constraints
        self.constraints = []

        # DEFINE THE CONSTANTS HERE
        #
        # Using the linear discrete forumlation we can now get A and B
        Ixx, Iyy, Izz, m = (
            self._getURDFParameter("ixx"),
            self._getURDFParameter("iyy"),
            self._getURDFParameter("izz"),
            self._getURDFParameter("m"),
        )
        assert Ixx is not None and Iyy is not None and Izz is not None and m is not None
        self.dynamics = MPCDynamics(Ixx, Iyy, Izz, m, g=g)
        #
        # Horizon steps
        self.horizon = horizon
        # Timestep value {s}
        self.dt = 0.01
        # Flight Envelope {s}
        self.T = 50
        # Target
        self.x_target = np.block(np.array([[10.0, 0.0, 5.0, np.zeros((9, 1))]]))

        # DEFINE THE WEIGHT MATRICES
        self.weight_input = 0.2 * np.eye(1)  # Weight on the input
        self.weight_tracking = 1.0 * np.eye(1)  # Weight on the tracking state

        self.reset()

    def reset(self):
        super().reset()

        self.x = cp.Variable((12, self.horizon + 1))
        self.u = cp.Variable((4, self.horizon))
        self.cost = 0
        self.constraints = []

    # def computeControlFromState(
    #     self,
    #     control_timestep,
    #     state,
    #     target_pos,
    #     target_rpy=np.zeros(3),
    #     target_vel=np.zeros(3),
    #     target_rpy_rates=np.zeros(3),
    # ):
    #     """
    #     In this function we compute the control commands
    #     """
    #
    #     return self._computeControl(
    #         control_timestep,
    #         state,
    #         target_pos,
    #         target_rpy,
    #         target_vel,
    #         target_rpy_rates,
    #     )

    def _computeControl(
        self,
        control_timestep,
        state,
    ):

        for horizon_step in range(self.horizon):
            self._build_constraints_iter(horizon_step, state)
            self._build_costs_iter(horizon_step, state)
        self._build_constraints_once(state)

        problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        problem.solve(solver=cp.OSQP)

        # delta_thrust, tau_roll, tau_pitch, tau_yaw = self.u[:, 0]

        rpms = self._preprocessAction(self.u[:, 0])

        return self.u[:, 0]

    def _preprocessAction(self, action):
        """
        action : u-input vector with elements delta_thrust, tau_roll, tau_pitch, tau_yaw
        """
        omegas = self.MIXER_MATRIX @ action.T
        rpms = self.KF * omegas**2
        rpms[0, :] += self.HOVER_RPM

        rpms = np.clip(rpms, self.MIN_RPM, self.MAX_RPM)

        return rpms

    def _build_constraints_iter(self, horizon_step, state):
        """
        Function to build the constraints at each iteration of the MPC horizon
        """
        self.constraints += [
            self.x[0:3, horizon_step] <= np.array([20.0, 10.0, 20.0]),
            self.x[0:3, horizon_step] >= np.array([-20.0, -10.0, -20.0]),
        ]

        # System dynamics constraint: x_{k+1} = A x_k + B u_k
        self.constraints += [
            self.x[:, horizon_step + 1]
            == self.dynamics.next_x(self.x[:, horizon_step], self.u[:, horizon_step])
        ]

    def _build_costs_iter(self, horizon_step, state):
        """
        Function to build the costs at each iteration of the MPC horizon
        """

        # End position tracking cost
        self.cost += self.weight_tracking[0, 0] * cp.square(
            self.x[0:3, horizon_step] - self.x_target[0:3]
        )
        # Force input cost term
        self.cost += self.weight_input[0, 0] * cp.sum_squares(self.u[:, horizon_step])

    def _build_constraints_once(self, state):
        """
        Function to build the contraints once at the end of MPC loop
        """

        unpacked_state = self._unpack_state(state)

        self.constraints += [self.x[:, 0] == unpacked_state]

    def _unpack_state(self, state):
        """
        This function must unpack the exact state that is used by the MPC controller
        Note thate BaseAviary organises the state params as such
        position : [0:3]
        quaternions : [3:7]
        rpy : [7:10]
        velocity : [10:13]
        angular_velocity : [13:16]
        last_rpm : [16:20]
        """
        res = np.block([[state[0:7], state[10:16]]])

        # Ensure that the shapes are the same
        assert res.shape == self.x.shape

        return res
