import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = X*0.5 + np.random.randn(N)
Y[-1] += 30
Y[-2] += 30

# plt.scatter(X, Y)
# plt.show()

# adding bias
X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()

# l2 penalty - out lambda from formula?
l2 = 1000.0
# np.eye - returns zero 2D  matrix with 1 on diagonal
w_map = np.linalg.solve(l2 * np.eye(2)+ X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='ml')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()