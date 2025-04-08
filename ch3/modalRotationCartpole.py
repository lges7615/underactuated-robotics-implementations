import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

# --- Define linearized A matrix at theta = pi for cart-pole subsystem ---
A = np.array([[0, 1], [-30, 0]])  # corresponds to g = 10 and -3g

# --- Compute eigenvectors and normalize ---
eigvals, eigvecs = eig(A)
v_real = np.real(eigvecs[:, 0])
v_imag = np.imag(eigvecs[:, 0])
v_real /= np.linalg.norm(v_real)
v_imag /= np.linalg.norm(v_imag)

# --- Set up the figure and axis ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-10, 10)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_title('Modal Rotation in $(\\theta, \dot{\\theta})$ Plane')
ax.grid(True)
ax.axis('equal')

# --- Plot modal directions (eigenvectors) ---
ax.quiver(
    0, 0, v_real[0], v_real[1], color='r', scale=1, scale_units='xy', angles='xy', label='Re{v}'
)
ax.quiver(
    0, 0, v_imag[0], v_imag[1], color='b', scale=1, scale_units='xy', angles='xy', label='Im{v}'
)
ax.legend()

# --- Create animated point that rotates in modal plane ---
(point,) = ax.plot([], [], 'ko', markersize=6)

# Time vector and circular motion in modal basis
t_vals = np.linspace(0, 2 * np.pi, 100)
trajectory = np.outer(np.cos(t_vals), v_real) + np.outer(np.sin(t_vals), v_imag)


# Initialization function
def init():
    point.set_data([], [])
    return (point,)


# Update function
def update(frame):
    x, y = trajectory[frame]
    point.set_data(x, y)
    return (point,)


# Animate
ani = animation.FuncAnimation(
    fig, update, frames=len(t_vals), init_func=init, blit=True, interval=50
)

plt.show()
