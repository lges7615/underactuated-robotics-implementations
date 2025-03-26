import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
q0 = 5  # Initial position
dq0 = 0  # Initial velocity
u_max = 1  # Control limit

ddq = u
dq = dq0 - t
q = q0 + dq0 * t - 0.5 * t ^ 2
q = -0.5 * dq ^ 2 + dq0 + 0.5 - dq

# Compute switching time analytically
t1 = np.sqrt(2 * q0)  # Time to switch from u=1 to u=-1
T = 2 * t1  # Total time to reach the origin

# Time vector
dt = 0.01
time = np.arange(0, T + dt, dt)

# Initialize state variables
q = np.zeros_like(time)
dq = np.zeros_like(time)
u = np.zeros_like(time)

q[0] = q0
dq[0] = dq0

# Simulate Bang-Bang control
for i in range(1, len(time)):
    t = time[i]
    if t < t1:
        u[i] = u_max  # Acceleration phase
    else:
        u[i] = -u_max  # Deceleration phase

    # State update using Euler integration
    dq[i] = dq[i - 1] + u[i] * dt
    q[i] = q[i - 1] + dq[i] * dt

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, q, label="Position q")
plt.ylabel("Position")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, dq, label="Velocity dq", color='r')
plt.ylabel("Velocity")
plt.legend()

plt.subplot(3, 1, 3)
plt.step(time, u, label="Control u", where='post', color='g')
plt.ylabel("Control Input")
plt.xlabel("Time")
plt.legend()

plt.suptitle("Minimum Time Bang-Bang Control for Double Integrator")
plt.show()
