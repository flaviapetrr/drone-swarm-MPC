import os

import numpy as np
from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.geometry import Pose

# Resolve paths independently of the current working directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")
BALL_MESH_PATH = os.path.join(REPO_ROOT, "mesh", "basketball.obj")


def obj_bounding_box(path):
    """Return (center, size) of the axis-aligned bounding box of an OBJ mesh."""
    verts = np.array(
        [line.split()[1:4] for line in open(path) if line.startswith("v ")], dtype=float
    )
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    return (vmin + vmax) / 2, vmax - vmin


sim_core = MultiRobotSimulationCore.from_yaml(CONFIG_PATH)

# The Blender export is not centered on the origin: shift the visual mesh so
# that its center coincides with the body frame (and the collision sphere)
ball_center, ball_size = obj_bounding_box(BALL_MESH_PATH)
ball_radius = float(ball_size.max()) / 2

# Static obstacle (mass = 0): not affected by gravity or contacts
ball = SimObject.from_mesh(
    visual_shape=ShapeParams(
        shape_type="mesh",
        mesh_path=BALL_MESH_PATH,
        rgba_color=[0.93, 0.47, 0.13, 1.0],  # basketball orange
        frame_pose=Pose.from_xyz(*(-ball_center)),
    ),
    collision_shape=ShapeParams(shape_type="sphere", radius=ball_radius),
    pose=Pose.from_xyz(-2, 0, ball_radius),  # resting on the ground
    sim_core=sim_core,
)

# Keep the simulation (and the GUI window) running until it is closed
sim_core.run_simulation()
