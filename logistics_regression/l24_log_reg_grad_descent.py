import numpy as np
import matplotlib.pyplot as plt


N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] + 2*np.ones((50, D))
X[50:, :] = X[50:, :] - 2*np.ones((50, D))

bias = np.ones((N, 1))

T = np.ones((N))
T[:50]=0

Xb = np.concatenate((bias, X), axis=1)

def sigmoid (z):
    return 1/(1+np.exp(-z))

def cross_entropy(T, Y):
    E = 0
    for i in range(T.shape[0]):
        if T[i]==1:
            E -=np.log(Y[i])
        else:
            E -=np.log(1 -Y[i])
    return E

# init weights
w = np.random.randn(D+1)
Y = sigmoid(Xb.dot(w))


# gradient descent
learning_rate =0.1
epochs =100
for i in range(epochs):
    w += learning_rate * Xb.T.dot(T-Y)
    if i %10 ==0:
        print("CE:",cross_entropy(T, Y))
        print("w:", w)
    Y = sigmoid(Xb.dot(w))

# visualizing
plt.scatter(X[: , 0], X[:, 1], c=T, s=100, alpha=0.5)

x_axis = np.linspace( -6, 6, 100)
# y = ax+b
y_axis = -(w[0] + x_axis*w[1])/w[2]
plt.plot(x_axis, y_axis)
plt.show()