import numpy as np
import matplotlib.pyplot as plt

N= 100
D = 2

X = np.random.randn(N, D)
X[:50, :] = X[:50, :] + 2* np.ones((50, 2))
X[50:, :] = X[50:, :] - 2* np.ones((50, 2))

T = np.ones((N))
T[:50] = 0

#bias
bias = np.ones((N,1))
Xb =  np.concatenate((bias, X), axis=1)

def sigma(z):
    return 1/(1+np.exp(-z))

def cross_entropy_error(T, Y):
    E = 0
    for i in range(T.shape[0]):
        if T[i]==1:
            E -= np.log(1-Y[i])
        else:
            E -= np.log(Y[i])
    return E

w = np.array([0,4,4])

# y = -x
# line_p1 = [-6, 6]
# line_p2 = [6, -6]
x_axis = np.linspace(-6, 6, 100)
y_axis = - x_axis

plt.scatter(X[:,0], X[:,1], c = T, s=100, alpha=0.5)
plt.plot(x_axis, y_axis)
plt.show()