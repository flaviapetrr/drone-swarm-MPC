from typing import TYPE_CHECKING

import numpy as np
import cvxpy as cp

# IMPORTANT
# The MPC has some basic constraints already in place. These are:
# - Initial Conditions constraint (x[0] = x_init = state_O)
# - Dynamics constraint (x[k+1] = Ax[k] + Bu[k])
# - Controller input constraints (basically the rpm is capped at a minimum and maximum)
#
# This means YOU DO NOT NEED TO IMPLEMENT THEM, if you need to change something lmk

if TYPE_CHECKING:
    from src.mpc_controller import MPCController


# Simple bounding box for our drone
# EVERY FUNCTION MUST HAVE ctrl and step as arguments
def position_bounds(ctrl: "MPCController", step: int):
    return [
        ctrl.pos(step) <= np.array([20.0, 10.0, 20.0]),
        ctrl.pos(step) >= np.array([-20.0, -10.0, -20.0]),
    ]


def input_cost(ctrl: "MPCController", step: int, weight: float = 1):
    return weight * cp.sum_squares(ctrl.u[:, step])


def wall_cost(ctrl: "MPCController", step: int, target_pos:np.ndarray, weights: np.ndarray):
    assert len(weights) == 3, "weights must be a 3-element array"
    return (weights[0] * cp.sum_squares(ctrl.pos(step) - target_pos)
                        + weights[1] * cp.sum_squares(ctrl.vel(step))
                        + weights[2] * cp.sum_squares(ctrl.ang_vel(step)))


def apply(ctrl: "MPCController", target_pos: np.ndarray, horizon: int) -> None:
    # Position limits at steps 0..horizon-1
    ctrl.add_constraint(position_bounds, steps=range(0, horizon))
    # Tracking at steps 0..horizon (includes the terminal state)
    ctrl.add_cost(
        lambda c, k: wall_cost(c, k, target_pos=target_pos, weights=np.array([10.0, 1, 0.1])),
        steps=range(0, horizon+1),
    )
    # Input effort at steps 0..horizon-1 (u has no column at horizon)
    ctrl.add_cost(input_cost, steps=range(0, horizon))
