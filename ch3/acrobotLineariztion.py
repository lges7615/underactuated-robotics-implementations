# Re-import necessary libraries after code execution state reset
import sympy as sp

# Define symbolic variables
theta1, theta2, dtheta1, dtheta2, u = sp.symbols('theta1 theta2 dtheta1 dtheta2 u')
m1, m2, l1, l2, lc1, lc2, I1, I2, g = sp.symbols('m1 m2 l1 l2 lc1 lc2 I1 I2 g')

# Define state vector
x = sp.Matrix([theta1, theta2, dtheta1, dtheta2])

# Define M(q)
c2 = sp.cos(theta2)
s2 = sp.sin(theta2)
M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * c2
M12 = I2 + m2 * l1 * lc2 * c2
M21 = M12
M22 = I2
M = sp.Matrix([[M11, M12], [M21, M22]])

# Define C(q, dq)*dq
C1 = -2 * m2 * l1 * lc2 * s2 * dtheta2 * dtheta1 - m2 * l1 * lc2 * s2 * dtheta2**2
C2 = m2 * l1 * lc2 * s2 * dtheta1**2
C = sp.Matrix([C1, C2])

# Define gravity vector
g1 = (m1 * lc1 + m2 * l1) * g * sp.cos(theta1) + m2 * lc2 * g * sp.cos(theta1 + theta2)
g2 = m2 * lc2 * g * sp.cos(theta1 + theta2)
G = sp.Matrix([g1, g2])

# Define B matrix
B_input = sp.Matrix([0, 1])

# Define state derivative
q = sp.Matrix([theta1, theta2])
dq = sp.Matrix([dtheta1, dtheta2])

ddq = M.inv() * (B_input * u - C - G)
dx = sp.Matrix([dtheta1, dtheta2, ddq[0], ddq[1]])

# Compute Jacobians
A = dx.jacobian(x)
B = dx.jacobian(sp.Matrix([u]))

# Define parameter values and upright configuration
subs_dict = {
    theta1: 0,
    theta2: 0,
    dtheta1: 0,
    dtheta2: 0,
    m1: 1,
    m2: 1,
    l1: 1,
    l2: 1,
    lc1: 0.5,
    lc2: 0.5,
    I1: 1,
    I2: 1,
    g: 9.81,
}

# Evaluate A and B matrices numerically
A_num = A.evalf(subs=subs_dict)
B_num = B.evalf(subs=subs_dict)

# A_num, B_num

print("A matrix (linearized):")
sp.pprint(A_num)  # 美觀格式輸出（SymPy 專用）
print("\nB matrix (linearized):")
sp.pprint(B_num)
