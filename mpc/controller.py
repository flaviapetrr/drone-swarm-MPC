"""Full-state MPC for the quadrotor: thrust and torque inputs, motor-space limits, soft tilt limit."""
import numpy as np
import cvxpy as cp

from mpc.drone_model import continuous_model, input_scale, discretize
from mpc.mixer import motor_limits

class MPC:
    def __init__(self, P, dt, N, max_tilt_deg, w_pos, w_vel, w_att, w_rate, w_u, w_eps):
        self.N = N
        A, B = continuous_model(P)
        self.u_scale = input_scale(P)
        Ad, Bd = discretize(A, B, dt, self.u_scale)
        F_map, f_hover, f_max = motor_limits(P, self.u_scale)
        xi_max = np.sin(np.radians(max_tilt_deg) / 2)

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
