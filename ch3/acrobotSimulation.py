import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Define system parameters
m1, m2 = 1.0, 1.0  # Masses of the links
l1, l2 = 1.0, 1.0  # Lengths of the links
lc1, lc2 = 0.5, 0.5  # Center of mass positions
I1, I2 = m1 * l1**2 / 3, m2 * l2**2 / 3  # Moments of inertia
g = -9.81  # Gravity acceleration


# System dynamics: θ1, θ2, θ̇1, θ̇2
def acrobot_dynamics(t, state, u=0):
    theta1, theta2, omega1, omega2 = state

    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)

    M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * c2
    M12 = I2 + m2 * l1 * lc2 * c2
    M21 = M12
    M22 = I2
    M = np.array([[M11, M12], [M21, M22]])

    C1 = -2 * m2 * l1 * lc2 * s2 * omega2 * omega1 - m2 * l1 * lc2 * s2 * omega2**2
    C2 = m2 * l1 * lc2 * s2 * omega1**2
    C = np.array([C1, C2])

    G1 = -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12)
    G2 = -m2 * g * lc2 * s12
    G = np.array([G1, G2])

    B = np.array([0, 1])  # Control torque acts only on the second joint
    tau = np.array([0, u])  # Applied torque

    # Solve for angular accelerations
    q_ddot = np.linalg.solve(M, tau - C - G)
    return [omega1, omega2, q_ddot[0], q_ddot[1]]


# Time settings
t_max = 20.0  # Simulation duration (seconds)
dt = 0.02  # Time step

# Initial state: [θ1, θ2, ω1, ω2]
initial_state = [np.pi / 2 + 0.5, 0.0, 0.0, 0.0]

# Solve the system
t_eval = np.arange(0, t_max, dt)
sol = solve_ivp(acrobot_dynamics, [0, t_max], initial_state, t_eval=t_eval, args=(0,))

# Extract solution
theta1_vals, theta2_vals = sol.y[0], sol.y[1]

# Convert to Cartesian coordinates
x1 = l1 * np.sin(theta1_vals)
y1 = -l1 * np.cos(theta1_vals)
x2 = x1 + l2 * np.sin(theta1_vals + theta2_vals)
y2 = y1 - l2 * np.cos(theta1_vals + theta2_vals)

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid()

(line,) = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(-1.8, 1.8, '', fontsize=12)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    time_text.set_text(f'Time: {t_eval[frame]:.2f} s')
    return line, time_text


ani = animation.FuncAnimation(
    fig, update, frames=len(t_eval), init_func=init, blit=True, interval=dt * 1000
)

plt.show()
