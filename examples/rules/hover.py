from typing import TYPE_CHECKING

import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from src.mpc_controller import MPCController


def position_bounds(ctrl: "MPCController", step: int):
    return [
        ctrl.pos(step) <= np.array([20.0, 10.0, 20.0]),
        ctrl.pos(step) >= np.array([-20.0, -10.0, -20.0]),
    ]


def distance_cost(
    ctrl: "MPCController", step: int, target_pos: np.ndarray, weight: float = 1.0
):
    return weight * cp.sum_squares(ctrl.pos(step) - target_pos)


def input_cost(ctrl: "MPCController", step: int, weight: float = 0.2):
    return weight * cp.sum_squares(ctrl.u[:, step])


def apply(ctrl: "MPCController", target_pos: np.ndarray, horizon: int) -> None:
    # Position limits at steps 0..horizon-1
    ctrl.add_constraint(position_bounds, steps=range(0, horizon))
    # Tracking at steps 0..horizon (includes the terminal state)
    ctrl.add_cost(
        lambda c, k: distance_cost(c, k, target_pos=target_pos),
        steps=range(0, horizon + 1),
    )
    # Input effort at steps 0..horizon-1 (u has no column at horizon)
    ctrl.add_cost(input_cost, steps=range(0, horizon))
