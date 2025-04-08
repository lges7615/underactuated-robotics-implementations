import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define Acrobot parameters
g = 9.81  # gravity (m/s^2)
m1 = 1.0  # mass of first link (kg)
m2 = 1.0  # mass of second link (kg)
l1 = 1.0  # length of first link (m)
l2 = 1.0  # length of second link (m)
lc1 = l1 / 2  # center of mass of first link (m)
lc2 = l2 / 2  # center of mass of second link (m)
I1 = m1 * l1**2 / 3  # moment of inertia of first link (kg.m^2)
I2 = m2 * l2**2 / 3  # moment of inertia of second link (kg.m^2)

# Constant control input
u_const = 0  # Fixed torque applied to the second joint


# Equations of motion for Acrobot
def acrobot_dynamics(t, state):
    """Computes the state derivatives for the Acrobot with a constant control input."""
    theta1, theta2, omega1, omega2 = state

    # Trigonometric shorthand
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)

    # Inertia matrix
    # M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    # M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    # M21 = M12
    # M22 = I2 + m2 * lc2**2

    M11 = I1 + I2 + m2 * lc1**2 + m2 * (l1**2 + 2 * l1 * lc2 * c2)
    M12 = I2 + m2 * (l1 * lc2 * c2)
    M21 = M12
    M22 = I2
    M = np.array([[M11, M12], [M21, M22]])

    # Coriolis and gravity terms
    C1 = -m2 * l1 * lc2 * s2 * (2 * omega1 * omega2 + omega2**2)
    C2 = m2 * l1 * lc2 * s2 * omega1**2
    G1 = (m1 * lc1 + m2 * l1) * g * s1 + m2 * lc2 * g * s12
    G2 = m2 * lc2 * g * s12

    # Control input (constant torque at the second joint)
    tau = np.array([0, u_const])

    # Solve for angular accelerations
    rhs = np.array([-C1 - G1, -C2 - G2]) + tau
    alpha = np.linalg.solve(M, rhs)

    return [omega1, omega2, alpha[0], alpha[1]]


# Simulation parameters
t_span = (0, 10)  # Time range (seconds)
# y0 = [np.pi/4, -np.pi/2, 0, 0]  # Initial state: [theta1, theta2, omega1, omega2]
y0 = [np.pi / 4, 0, 0, 0]  # Initial state: [theta1, theta2, omega1, omega2]

# Solve the system
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(acrobot_dynamics, t_span, y0, t_eval=t_eval)

# Compute joint positions
x1 = l1 * np.sin(sol.y[0])
y1 = -l1 * np.cos(sol.y[0])
x2 = x1 + l2 * np.sin(sol.y[0] + sol.y[1])
y2 = y1 - l2 * np.cos(sol.y[0] + sol.y[1])

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Acrobot Animation")
ax.grid()

(line,) = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return (line,)


ani = animation.FuncAnimation(
    fig, update, frames=len(sol.t), init_func=init, blit=True, interval=20
)
plt.show()
