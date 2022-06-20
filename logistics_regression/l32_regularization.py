import numpy as np
import matplotlib.pyplot as plt

# fat matrix
N = 50
D = 50

X = (np.random.randn(N, D) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = X.dot(true_w)

plt.plot(X)
plt.scatter(X, Y)
plt.show()
