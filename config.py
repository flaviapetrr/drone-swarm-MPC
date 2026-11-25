"""All tunable settings in one place."""
import numpy as np

# ---- MPC
DT_MPC = 0.05          # [s] length of one prediction step inside the MPC
N_MPC = 40             # prediction steps -> horizon = N_MPC * DT_MPC = 2 s
MAX_TILT_DEG = 15.0    # soft limit; the linear model is only valid near hover

MPC_WEIGHTS = dict(
    w_pos=10.,         # distance to the target position
    w_vel=1.,          # velocity (acts as damping)
    w_att=1.,          # tilt
    w_rate=0.1,        # rotation rates
    w_u=1.,            # input effort (on the scaled inputs)
    w_eps=1e3,         # price of violating the tilt limit (must be much larger than the rest)
)

# ---- Mission
HOVER_POS = np.array([0.0, 0.0, 1.0])   # [m] phase 1: climb to here and hover
REF_POS = np.array([1.0, 0.5, 1.0])     # [m] phase 2: go here after hovering
SETTLE_TOL = 0.05                       # [m] closer than this to the hover point ...
SETTLE_VEL = 0.1                        # [m/s] ... and slower than this ...
SETTLE_SEC = 1.0                        # [s] ... continuously for this long = "arrived"

# ---- Simulation
SIM_FREQ_HZ = 240                       # [Hz] physics rate
CTRL_FREQ_HZ = 48                       # [Hz] how often the MPC runs
DURATION_SEC = 12                       # [s] length of the run
INIT_POS = np.array([0.0, 0.0, 0.1])    # [m] start position, 10 cm above the ground