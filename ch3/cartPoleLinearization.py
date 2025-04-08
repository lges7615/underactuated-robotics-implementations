import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

# Update parameters
mc = 1
mp = 1
l = 1
g = 10

# Update A matrix based on new parameters
A_updated = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, mp * g / mc, 0, 0],
        [0, g * (mc + mp) / (l * mc), 0, 0],
    ]
)

# Compute new eigenvalues and eigenvectors
eigenvalues_updated, eigenvectors_updated = eig(A_updated)
eigenvectors_updated_real = np.real(eigenvectors_updated)

eigenvalues_updated, eigenvectors_updated_real
print("eigen values:", eigenvalues_updated)
print("eigen vector:", eigenvectors_updated_real)


# Choose only the eigenvectors associated with nonzero eigenvalues (unstable/stable modes)
# These are the third and fourth columns
v1 = eigenvectors_updated_real[:, 2]  # lambda = +4.4721 (unstable)
v2 = eigenvectors_updated_real[:, 3]  # lambda = -4.4721 (stable)

# Normalize for plotting
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# We project onto the (theta, dtheta) plane (index 1 and 3)
fig, ax = plt.subplots()
ax.quiver(
    0,
    0,
    v1[1],
    v1[3],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="r",
    label="Unstable Mode",
)
ax.quiver(
    0,
    0,
    v2[1],
    v2[3],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="b",
    label="Stable Mode",
)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\dot{\theta}$")
ax.set_title("Modal Directions in $(\\theta, \dot{\\theta})$ Plane")
ax.grid(True)
ax.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
