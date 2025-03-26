import numpy as np
import matplotlib.pyplot as plt

# Reference: Underactuated textbook Example 7.2 The minimum time problem for the double integrator


# Define the cost-to-go function J(q, dq) for different control inputs u
def J_x_positive(q, dq):
    return 2 * np.sqrt(0.5 * dq**2 - q) - dq


def J_x_negative(q, dq):
    return dq + 2 * np.sqrt(0.5 * dq**2 + q)


# Create meshgrid for q and dq
q_vals = np.linspace(-10, 10, 100)
dq_vals = np.linspace(-4, 4, 100)
Q, DQ = np.meshgrid(q_vals, dq_vals)

# Create an empty array to store the computed y values with the same shape as Q
y = np.empty_like(Q, dtype=float)

# Create boolean masks for different conditions:
# Mask 1: ((dq < 0) and (q <= 0.5 * dq**2)) or ((dq >= 0) and (q < -0.5 * dq**2))
mask1 = ((DQ < 0) & (Q <= 0.5 * (DQ**2))) | ((DQ >= 0) & (Q < -0.5 * (DQ**2)))

# Mask 2: (q == 0) and (dq == 0) (using np.isclose for floating point comparison)
mask2 = np.isclose(Q, 0) & np.isclose(DQ, 0)

# Mask 3: all other cases
mask3 = ~(mask1 | mask2)

# Compute y values based on the conditions:
y[mask1] = J_x_positive(Q[mask1], DQ[mask1])
y[mask2] = 0
y[mask3] = J_x_negative(Q[mask3], DQ[mask3])

# Plot the results using a filled contour plot
plt.figure()
contour = plt.contourf(Q, DQ, y, 20, cmap='jet')
plt.colorbar(contour)
plt.xlabel('q')
plt.ylabel('dq')
plt.title('Contour Plot: Conditional y Calculation')
plt.show()
