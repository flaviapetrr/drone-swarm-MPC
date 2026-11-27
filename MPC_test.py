import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

dt = 1/48
N = 30                              # horizon length (steps)
Ad = np.array([[1, dt], [0, 1]])    # discrete-time dynamics matrix                    
Bd = np.array([[0.5*dt**2], [dt]])  # discrete-time input matrix

opti = ca.Opti()
X = opti.variable(2, N+1)    # states over the horizon: [p; v] at each step
U = opti.variable(1, N)      # accelerations over the horizon
x0 = opti.parameter(2)       # current measured state (set at every step)
p_ref = opti.parameter()     # target position

opti.subject_to(X[:, 0] == x0)

for k in range(N):
    # constrain X[:, k+1] to follow the model from X[:, k] and U[:, k]
    # dynamics 
    opti.subject_to(X[:, k+1] == Ad @ X[:, k] + Bd @ U[:, k])

# limits, e.g. |U| <= a_max and |v| <= v_max (pick numbers, e.g. 2 m/s^2 and 1 m/s)
opti.subject_to(opti.bounded(-2, U, 2))
opti.subject_to(opti.bounded(-1, X[1, :], 1))

# cost = sum over the horizon of (position error)^2 + small weight * (acceleration)^2
cost = 0
cost += ca.sumsqr(X[0, :] - p_ref) + 0.01 * ca.sumsqr(U) + 0.01 * ca.sumsqr(X[1, :]) + 10 * ca.sumsqr(X[0, N] - p_ref) + ca.sumsqr(X[1, N])
opti.minimize(cost)

opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

x = np.array([0.0, 0.0])
p_history = []

for T in range(200):
    opti.set_value(x0, x)
    opti.set_value(p_ref, 1.0)  # target position is 1.0 m
    sol = opti.solve()
    u_opt = sol.value(U[0, 0])  # optimal acceleration for the first step
    x = Ad @ x + Bd.flatten() * u_opt      # update state with the optimal control
    p_history.append(x[0])       # store position history
    print(f"Step {T}: Position = {x[0]:.3f}, Velocity = {x[1]:.3f}, Acceleration = {u_opt:.3f}")

plt.plot(np.arange(len(p_history))*dt, p_history)
plt.show()