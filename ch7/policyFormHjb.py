import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Initialize optimal policy storage
U_star = np.zeros_like(J)

# Extract optimal policy from value function
for i in range(len(q_vals)):
    for j in range(len(dq_vals)):
        q = q_vals[i]
        dq = dq_vals[j]
        costs = []
        for u in u_vals:
            q_next = q + dq * dt
            dq_next = dq + u * dt

            # Interpolate J at next state
            if -5 <= q_next <= 5 and -5 <= dq_next <= 5:
                i_next = np.searchsorted(q_vals, q_next)
                j_next = np.searchsorted(dq_vals, dq_next)
                i_next = np.clip(i_next, 0, len(q_vals) - 1)
                j_next = np.clip(j_next, 0, len(dq_vals) - 1)
                J_next = J[i_next, j_next]
            else:
                J_next = 1e6  # large cost for going out of bounds

            l = q**2 + dq**2 + u**2
            cost = l + gamma * J_next
            costs.append(cost)

        U_star[i, j] = u_vals[np.argmin(costs)]

# Plot optimal policy as vector field
plt.figure(figsize=(8, 6))
plt.quiver(
    Q, DQ, np.zeros_like(U_star), U_star, angles='xy', scale_units='xy', scale=5, color='blue'
)
plt.xlabel('Position q')
plt.ylabel('Velocity dq')
plt.title('Optimal Policy (Control Input u*) from HJB Solution')
plt.grid(True)
plt.show()
