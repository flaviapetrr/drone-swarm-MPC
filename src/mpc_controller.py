from collections.abc import Callable, Iterable, Sequence

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import cvxpy as cp
import numpy as np

ConstraintRule = Callable[
    ["MPCController", int], cp.Constraint | Sequence[cp.Constraint]
]
CostRule = Callable[["MPCController", int], cp.Expression]


class MPCDynamics:
    """Linearized model with state [pos, q_xyz, vel, pqr]."""

    def __init__(
        self,
        Ixx: float,
        Iyy: float,
        Izz: float,
        mass: float,
        g: float = 9.81,
        dt: float = 0.01,
        dtype=np.float64,
    ) -> None:
        z33 = np.zeros((3, 3), dtype=dtype)
        z31 = np.zeros((3, 1), dtype=dtype)
        i33 = np.eye(3, dtype=dtype)
        e3 = np.array([[0.0], [0.0], [1.0]], dtype=dtype)

        J_inv = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz]).astype(dtype)

        G = np.array(
            [
                [0.0, 2.0 * g, 0.0],
                [-2.0 * g, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=dtype,
        )

        # Continuous state order: [position, q_xyz, velocity, pqr].
        self.A_c = np.block(
            [
                [z33, z33, i33, z33],
                [z33, z33, z33, 0.5 * i33],
                [z33, G, z33, z33],
                [z33, z33, z33, z33],
            ]
        )

        # Input order: [delta_thrust, roll, pitch, yaw].
        self.B_c = np.block(
            [
                [z31, z33],
                [z31, z33],
                [e3 / mass, z33],
                [z31, J_inv],
            ]
        )

        self.dt = dt
        self.A_d = np.eye(12, dtype=dtype) + dt * self.A_c
        self.B_d = dt * self.B_c

    def next_x(self, x, u):
        return self.A_d @ x + self.B_d @ u


class MPCController(BaseControl):
    """MPC controller returning four motor RPMs for one drone."""

    def __init__(
        self,
        drone_model: DroneModel,
        horizon: int,
        g: float = 9.8,
        dt: float = 0.1,
        min_rpm: float = 0.0,
        max_rpm: float = 0.0,
        hover_rpm: float = 0.0,
    ) -> None:
        super().__init__(drone_model=drone_model, g=g)

        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not (0.0 <= min_rpm < hover_rpm < max_rpm):
            raise ValueError(
                "Require 0 <= min_rpm < hover_rpm < max_rpm; "
                "pass the limits from the aviary."
            )

        arm = self._getURDFParameter("arm")
        assert arm is not None
        Ixx = self._getURDFParameter("ixx")
        assert Ixx is not None
        Iyy = self._getURDFParameter("iyy")
        assert Iyy is not None
        Izz = self._getURDFParameter("izz")
        assert Izz is not None
        mass = self._getURDFParameter("m")
        assert mass is not None

        self.MIN_RPM = float(min_rpm)
        self.MAX_RPM = float(max_rpm)
        self.HOVER_RPM = float(hover_rpm)
        self.horizon = int(horizon)
        self.dt = float(dt)

        assert self.KF is not None and self.KM is not None

        yaw_force_ratio = self.KM / self.KF

        if self.DRONE_MODEL == DroneModel.CF2X:
            lever = arm / np.sqrt(2.0)
            # Rows map motor forces to [total thrust, roll_x, pitch_y, yaw_z].
            self.WRENCH_FROM_MOTOR_FORCE = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [-lever, -lever, lever, lever],
                    [-lever, lever, lever, -lever],
                    [
                        -yaw_force_ratio,
                        yaw_force_ratio,
                        -yaw_force_ratio,
                        yaw_force_ratio,
                    ],
                ]
            )
        elif self.DRONE_MODEL == DroneModel.CF2P:

            # Rows map motor forces to [total thrust, roll_x, pitch_y, yaw_z].
            self.WRENCH_FROM_MOTOR_FORCE = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, arm, 0.0, -arm],
                    [-arm, 0.0, arm, 0.0],
                    [
                        -yaw_force_ratio,
                        yaw_force_ratio,
                        -yaw_force_ratio,
                        yaw_force_ratio,
                    ],
                ]
            )
        else:
            raise ValueError(f"Unsupported drone model: {self.DRONE_MODEL}")

        self.MOTOR_FORCE_FROM_WRENCH = np.linalg.inv(self.WRENCH_FROM_MOTOR_FORCE)
        self.HOVER_FORCE_PER_MOTOR = self.KF * self.HOVER_RPM**2
        self.MIN_MOTOR_FORCE = self.KF * self.MIN_RPM**2
        self.MAX_MOTOR_FORCE = self.KF * self.MAX_RPM**2

        self.dynamics = MPCDynamics(
            Ixx=Ixx,
            Iyy=Iyy,
            Izz=Izz,
            mass=mass,
            g=g,
            dt=dt,
        )

        # [position, q_xyz, velocity, pqr].
        self._x = cp.Variable((12, self.horizon + 1))
        # [delta_thrust, roll_torque, pitch_torque, yaw_torque]
        self._u = cp.Variable((4, self.horizon))

        self._constraint_rules: list[tuple[ConstraintRule, list[int]]] = []
        self._cost_rules: list[tuple[CostRule, list[int]]] = []

        self.reset()

    def reset(self):
        super().reset()

    def computeControlFromState(
        self,
        control_timestep,
        state,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    ):
        return self.computeControl(
            control_timestep,
            *self._unpack_state(state),
            target_pos,
            target_rpy,
            target_vel,
            target_rpy_rates,
        )

    def computeControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        cur_ang_vel,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    ):
        if not np.isclose(control_timestep, self.dt):
            raise ValueError(
                f"Controller dt ({self.dt}) does not match control_timestep "
                f"({control_timestep})"
            )

        self.cost: cp.Expression = cp.Constant(0.0)
        self.constraints: list[cp.Constraint] = []

        self._add_core_constraints(cur_pos, cur_quat, cur_vel, cur_ang_vel)
        self._apply_rules()

        problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC solve failed: {problem.status}")
        if self.u[:, 0].value is None:
            raise RuntimeError("MPC solver returned no control solution")

        if self.x.value is None:
            raise RuntimeError("MPC solver returned no state prediction")

        # Shape: (3, horizon + 1), columns are predicted positions.
        self.predicted_positions = np.array(self.x.value[0:3, :], copy=True)

        return self._preprocessAction(self.u[:, 0].value)

    def _preprocessAction(self, action):
        """Allocate [delta thrust, roll, pitch, yaw] and convert to RPM."""
        action = np.asarray(action, dtype=float).reshape(4)

        motor_force_delta = self.MOTOR_FORCE_FROM_WRENCH @ action
        motor_forces = self.HOVER_FORCE_PER_MOTOR + motor_force_delta
        motor_forces = np.clip(
            motor_forces,
            self.MIN_MOTOR_FORCE,
            self.MAX_MOTOR_FORCE,
        )
        return np.sqrt(motor_forces / self.KF)

    # Core constraints that every scenario relies on
    def _add_core_constraints(self, cur_pos, cur_quat, cur_vel, cur_ang_vel):
        x0 = np.concatenate(
            [
                np.asarray(cur_pos, dtype=float).reshape(3),
                np.asarray(cur_quat, dtype=float).reshape(3),
                np.asarray(cur_vel, dtype=float).reshape(3),
                np.asarray(cur_ang_vel, dtype=float).reshape(3),
            ]
        )
        self.constraints.append(self._x[:, 0] == x0)

        for k in range(self.horizon):
            self.constraints.append(
                self._x[:, k + 1] == self.dynamics.next_x(self._x[:, k], self._u[:, k])
            )
            motor_forces = (
                self.HOVER_FORCE_PER_MOTOR
                + self.MOTOR_FORCE_FROM_WRENCH @ self._u[:, k]
            )
            self.constraints += [
                motor_forces >= self.MIN_MOTOR_FORCE,
                motor_forces <= self.MAX_MOTOR_FORCE,
            ]

    def _unpack_state(self, state):
        # BaseAviary state: pos[0:3], quaternion[3:7], rpy[7:10],
        # velocity[10:13], angular velocity[13:16], motor RPM[16:20].
        return state[0:3], state[3:6], state[10:13], state[13:16]

    def _required_kwarg(self, kwargs, name):
        if name not in kwargs or kwargs[name] is None:
            raise ValueError(f"Missing required keyword argument: {name}")
        return kwargs[name]

    # Properties to protect important variables from being manually changed
    @property
    def x(self) -> cp.Variable:
        return self._x

    @property
    def u(self) -> cp.Variable:
        return self._u

    # Named accessors for the various state stuff
    def pos(self, k: int) -> cp.Expression:
        return self._x[0:3, k]

    def quat(self, k: int) -> cp.Expression:
        return self._x[3:6, k]

    def vel(self, k: int) -> cp.Expression:
        return self._x[6:9, k]

    def ang_vel(self, k: int) -> cp.Expression:
        return self._x[9:12, k]

    # Functions for adding constraints and costs
    def add_constraint(self, rule: ConstraintRule, steps: Iterable[int]) -> None:
        steps = list(steps)
        self._check_steps(steps, max_k=self.horizon)
        self._constraint_rules.append((rule, steps))

    def add_cost(self, rule: CostRule, steps: Iterable[int]) -> None:
        steps = list(steps)
        self._check_steps(steps, max_k=self.horizon)
        self._cost_rules.append((rule, steps))

    # Helper functions
    def _check_steps(self, steps: list[int], max_k: int) -> None:
        for k in steps:
            if not 0 <= k <= max_k:
                raise ValueError(f"step {k} is outside 0..{max_k}")

    def _apply_rules(self) -> None:
        for rule, steps in self._constraint_rules:
            for k in steps:
                result = rule(self, k)
                items = [result] if isinstance(result, cp.Constraint) else list(result)
                for c in items:
                    if not c.variables():
                        raise ValueError(
                            f"constraint at step {k} has no decision variables"
                        )
                    self.constraints.append(c)
        for rule, steps in self._cost_rules:
            for k in steps:
                self.cost = self.cost + rule(self, k)
