import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))

X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(int(N/2)) + R_outer
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
X = np.concatenate([X_inner, X_outer])

T = np.array([0]*(int(N/2)) + [1]*(int(N/2)))

plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()
