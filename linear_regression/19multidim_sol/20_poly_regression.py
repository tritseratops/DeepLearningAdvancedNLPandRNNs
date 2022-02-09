# original file: lr_poly.py

import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open("data_poly.csv"):
    x, y = line.split(',')
    x = float(x)
    y = float(y)
    X.append([1, x, x*x])
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# plot original data
# plt.scatter(X[:,1],Y)
# plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot original data
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

# calculate how good model is computing r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("The r-squared is:", r2)

