import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define Cart-Pole parameters
g = 9.81  # gravity (m/s^2)
m_cart = 1.0  # mass of the cart (kg)
m_pole = 0.2  # mass of the pole (kg)
l_pole = 0.5  # length of the pole (m)
I_pole = m_pole * l_pole**2 / 3  # moment of inertia of the pole (kg.m^2)

# Constant control input
u_const = 1  # Fixed force applied to the cart

# Equations of motion for Cart-Pole
def cart_pole_dynamics(t, state):
    """ Computes the state derivatives for the Cart-Pole system with a constant control input."""
    x, theta, x_dot, theta_dot = state

    # Trigonometric shorthand
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Mass matrix components
    M11 = m_cart + m_pole
    M12 = m_pole * l_pole * cos_theta
    M21 = M12
    M22 = I_pole + m_pole * l_pole**2
    M = np.array([[M11, M12], [M21, M22]])

    # Coriolis and gravity terms
    C1 = -m_pole * l_pole * sin_theta * theta_dot**2
    C2 = -m_pole * g * l_pole * sin_theta

    # Control input (force on the cart)
    tau = np.array([u_const, 0])

    # Solve for accelerations
    rhs = np.array([C1, C2]) + tau
    accels = np.linalg.solve(M, rhs)

    return [x_dot, theta_dot, accels[0], accels[1]]

# Simulation parameters
t_span = (0, 10)  # Time range (seconds)
y0 = [0, np.pi/3*2, 0, 0]  # Initial state: [x, theta, x_dot, theta_dot]

# Solve the system
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(cart_pole_dynamics, t_span, y0, t_eval=t_eval)

# Compute positions
cart_x = sol.y[0]
pole_x = cart_x + l_pole * np.sin(sol.y[1])
pole_y = -l_pole * np.cos(sol.y[1])

# Animation
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Cart-Pole Animation")
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([cart_x[frame], pole_x[frame]], [0, pole_y[frame]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True, interval=20)
plt.show()
