import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# System parameters
m_c = 1.0  # Cart mass (kg)
m_p = 0.1  # Pole mass (kg)
l = 0.5  # Pole length (m)
g = 9.81  # Gravity (m/s^2)

# Define the Cart-Pole dynamics using the symmetric Coriolis formulation:
#   M(q) ddot{q} + C_sym(q, dot{q}) dot{q} + G(q) = B u,
# where u = f_x (the horizontal force on the cart).
#
# With:
#   M(q) = [[m_c+m_p, m_p*l*cos(theta)],
#           [m_p*l*cos(theta), m_p*l^2]]
#
#   C_sym(q, dot{q}) = [[0, -m_p*l*sin(theta)*theta_dot],
#                        [-m_p*l*sin(theta)*theta_dot, 0]]
#
#   G(q) = [0, -m_p*g*l*sin(theta)]^T
#
# The resulting equations are:
#
#   (m_c+m_p)*x_ddot + m_p*l*cos(theta)*theta_ddot - m_p*l*sin(theta)*theta_dot**2 = f_x
#   m_p*l*cos(theta)*x_ddot + m_p*l**2*theta_ddot - m_p*l*sin(theta)*theta_dot*x_dot = -m_p*g*l*sin(theta)
#
# We solve these equations for x_ddot and theta_ddot.


def cart_pole_dynamics(t, state, f_x):
    x, theta, x_dot, theta_dot = state

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Compute common term: denominator = m_p*l^2*(m_c + m_p*sin^2(theta))
    denominator = m_p * l**2 * (m_c + m_p * sin_theta**2)

    # Equation (1): (m_c+m_p)*x_ddot + m_p*l*cos(theta)*theta_ddot = f_x + m_p*l*sin(theta)*theta_dot**2
    # Equation (2): m_p*l*cos(theta)*x_ddot + m_p*l**2*theta_ddot = -m_p*g*l*sin(theta) + m_p*l*sin(theta)*theta_dot*x_dot
    #
    # Solve the 2x2 linear system:
    #   A * x_ddot + B * theta_ddot = R1
    #   C * x_ddot + D * theta_ddot = R2
    #
    # with:
    A = m_c + m_p
    B = m_p * l * cos_theta
    C = m_p * l * cos_theta
    D = m_p * l**2
    R1 = f_x + m_p * l * sin_theta * theta_dot**2
    R2 = -m_p * g * l * sin_theta + m_p * l * sin_theta * theta_dot * x_dot

    # Determinant of M
    det_M = A * D - B * C  # = m_p*l**2*(m_c + m_p*sin_theta**2)

    # Solve for x_ddot and theta_ddot
    x_ddot = (D * R1 - B * R2) / det_M
    theta_ddot = (-C * R1 + A * R2) / det_M

    return [x_dot, theta_dot, x_ddot, theta_ddot]


# Simulation parameters
t_max = 10.0  # Total simulation time (s)
dt = 0.02  # Time step (s)
t_eval = np.arange(0, t_max, dt)

# Initial conditions: [x, theta, x_dot, theta_dot]
# (Slight deviation from the upright position for the pole)
initial_state = [0.0, np.pi - 0.1, 0.0, 0.0]

# For this simulation, we set the applied force f_x = 0 (passive dynamics)
f_x = 0.0

# Solve the ODE
sol = solve_ivp(cart_pole_dynamics, [0, t_max], initial_state, t_eval=t_eval, args=(f_x,))

# Extract the solution
x_vals = sol.y[0]
theta_vals = sol.y[1]

# For animation: compute the coordinates for the cart and pole.
cart_width = 0.4
cart_height = 0.2
# For visualization, we choose a pole length that is twice the parameter l.
visual_pole_length = 2 * l
# Set the vertical coordinate for the cart (assume its center is at y = 0)
cart_y = -cart_height / 2

# The pole is attached to the cart at (x, cart_y) and its tip is computed by:
pole_tip_x = x_vals + visual_pole_length * np.sin(theta_vals)
pole_tip_y = cart_y - visual_pole_length * np.cos(theta_vals)

# Create the animation
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim([-2, 2])
ax.set_ylim([-1.2, 1.2])
ax.set_xlabel("Cart Position (m)")
ax.set_ylabel("Pole Position (m)")

# Draw the cart as a rectangle and the pole as a line
cart_patch = plt.Rectangle((x_vals[0] - cart_width / 2, cart_y), cart_width, cart_height, fc='b')
(pole_line,) = ax.plot([x_vals[0], pole_tip_x[0]], [cart_y, pole_tip_y[0]], 'r-', lw=2)
ax.add_patch(cart_patch)


def update(frame):
    cart_patch.set_x(x_vals[frame] - cart_width / 2)
    pole_line.set_xdata([x_vals[frame], pole_tip_x[frame]])
    pole_line.set_ydata([cart_y, pole_tip_y[frame]])
    return cart_patch, pole_line


ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=dt * 1000, blit=True)
plt.show()
