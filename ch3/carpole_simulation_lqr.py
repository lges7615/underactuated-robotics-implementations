import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Define Cart-Pole parameters
g = 9.81  # gravity (m/s^2)
m_cart = 1.0  # mass of the cart (kg)
m_pole = 0.2  # mass of the pole (kg)
l_pole = 0.5  # length of the pole (m)
I_pole = m_pole * l_pole**2 / 3  # moment of inertia of the pole (kg.m^2)

# Compute linearized state-space matrices A, B at the upright position
A = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -m_pole * g / m_cart, 0],
        [0, 0, 0, 1],
        [0, 0, (m_cart + m_pole) * g / (l_pole * m_cart), 0],
    ]
)

B = np.array([[0], [1 / m_cart], [0], [-1 / (l_pole * m_cart)]])

# Define LQR cost matrices
Q = np.diag([10, 1, 10, 1])  # State error cost
R = np.array([[0.01]])  # Control effort cost

# Solve the continuous-time algebraic Riccati equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute the LQR gain
K = np.linalg.inv(R) @ B.T @ P
print("LQR Gain Matrix K:", K)


# Define Cart-Pole dynamics with LQR control
def cart_pole_lqr(t, state):
    x, x_dot, theta, theta_dot = state
    u = float(-K @ np.array([x, x_dot, theta, theta_dot]))  # Ensure u is scalar
    u = np.clip(u, -10, 10)  # Limit control input
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Compute accelerations
    M11 = m_cart + m_pole
    M12 = m_pole * l_pole * cos_theta
    M21 = M12
    M22 = I_pole + m_pole * l_pole**2
    M = np.array([[M11, M12], [M21, M22]])

    C1 = -m_pole * l_pole * sin_theta * theta_dot**2 + u
    C2 = -m_pole * g * l_pole * sin_theta

    rhs = np.array([C1, C2], dtype=np.float64)  # Ensure correct dtype
    accels = np.linalg.solve(M, rhs)

    return [x_dot, accels[0], theta_dot, accels[1]]


# Simulation parameters
t_span = (0, 10)
y0 = [0, 0, np.pi * 6, 0]  # Initial state: [x, x_dot, theta, theta_dot]

t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(cart_pole_lqr, t_span, y0, t_eval=t_eval)

# Compute Cart-Pole positions
cart_x = sol.y[0]
pole_x = cart_x + l_pole * np.sin(sol.y[2])
pole_y = -l_pole * np.cos(sol.y[2])

# Animation
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-20, 20)
ax.set_ylim(-1, 1)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Cart-Pole LQR Animation")
ax.grid()

(line,) = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    line.set_data([cart_x[frame], pole_x[frame]], [0, pole_y[frame]])
    return (line,)


ani = animation.FuncAnimation(
    fig, update, frames=len(sol.t), init_func=init, blit=True, interval=20
)
plt.show()
