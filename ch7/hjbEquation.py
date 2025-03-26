# Re-import necessary libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# Reconstruct the environment with reduced settings
q_vals = np.linspace(-5, 5, 50)
dq_vals = np.linspace(-5, 5, 50)
Q, DQ = np.meshgrid(q_vals, dq_vals)
dt = 0.1
gamma = 0.95
u_vals = np.linspace(-1, 1, 11)
max_iter = 30

# Initialize cost-to-go function
J = np.zeros_like(Q)

# Value iteration
for iteration in range(max_iter):
    J_new = np.copy(J)
    for i in range(len(q_vals)):
        for j in range(len(dq_vals)):
            q = q_vals[i]
            dq = dq_vals[j]
            costs = []
            for u in u_vals:
                q_next = q + dq * dt
                dq_next = dq + u * dt
                if -5 <= q_next <= 5 and -5 <= dq_next <= 5:
                    i_next = np.searchsorted(q_vals, q_next)
                    j_next = np.searchsorted(dq_vals, dq_next)
                    i_next = np.clip(i_next, 0, len(q_vals) - 1)
                    j_next = np.clip(j_next, 0, len(dq_vals) - 1)
                    J_next = J[i_next, j_next]
                else:
                    J_next = 1e6
                l = q**2 + dq**2 + u**2
                costs.append(l + gamma * J_next)
            J_new[i, j] = min(costs)
    J = J_new

# Extract optimal policy
U_star = np.zeros_like(J)
for i in range(len(q_vals)):
    for j in range(len(dq_vals)):
        q = q_vals[i]
        dq = dq_vals[j]
        costs = []
        for u in u_vals:
            q_next = q + dq * dt
            dq_next = dq + u * dt
            if -5 <= q_next <= 5 and -5 <= dq_next <= 5:
                i_next = np.searchsorted(q_vals, q_next)
                j_next = np.searchsorted(dq_vals, dq_next)
                i_next = np.clip(i_next, 0, len(q_vals) - 1)
                j_next = np.clip(j_next, 0, len(dq_vals) - 1)
                J_next = J[i_next, j_next]
            else:
                J_next = 1e6
            l = q**2 + dq**2 + u**2
            costs.append(l + gamma * J_next)
        U_star[i, j] = u_vals[
            np.argmin(costs)
        ]  # Find the control vaule by finding the index of the minimum cost

# Simulate a trajectory under the optimal policy
initial_state = np.array([4.0, -3.0])
trajectory = [initial_state]
x = initial_state.copy()
for t in range(100):
    q, dq = x
    if not (-5 <= q <= 5 and -5 <= dq <= 5):
        break
    i = np.searchsorted(q_vals, q)
    j = np.searchsorted(dq_vals, dq)
    i = np.clip(i, 0, len(q_vals) - 1)
    j = np.clip(j, 0, len(dq_vals) - 1)
    u = U_star[i, j]
    q_next = q + dq * dt
    dq_next = dq + u * dt
    x = np.array([q_next, dq_next])
    trajectory.append(x)
trajectory = np.array(trajectory)

# Setup animation plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(
    Q,
    DQ,
    np.zeros_like(U_star),
    U_star,
    angles='xy',
    scale_units='xy',
    scale=5,
    color='lightgray',
    alpha=0.5,
)
ax.plot(0, 0, 'ko', label='Target (0, 0)')
(trajectory_line,) = ax.plot([], [], 'r.-', label='Optimal Trajectory')
(point,) = ax.plot([], [], 'ro')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('Position q')
ax.set_ylabel('Velocity dq')
ax.set_title('System Trajectory under Optimal Policy (Animation)')
ax.legend()
ax.grid(True)


def update(frame):
    if frame >= len(trajectory):
        return trajectory_line, point
    traj = trajectory[: frame + 1]
    trajectory_line.set_data(traj[:, 0], traj[:, 1])
    point.set_data([traj[-1, 0]], [traj[-1, 1]])  # Notice: must be list
    return trajectory_line, point


ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=True)

plt.show()
