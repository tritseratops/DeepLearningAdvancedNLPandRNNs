import numpy as np
from keras.backend import sigmoid

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] + 2*np.ones((50, D))
X[50:, :] = X[50, :] - 2*np.ones((50, D))

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
            E -=np.log(1- Y[i])
        else:
            E-=np.log(Y[i])
    return E

# init weights
w = np.random.randn(D+1)
Y = sigmoid(Xb.dot(w))


# gradient descent
learning_rate =0.1
epochs =100
for i in range(epochs):
    w += learning_rate * Xb.T.dot(Y-T)
    if i %10 ==0:
        print("CE:",cross_entropy(T, Y))
        print("w:", w)
    Y = sigmoid(Xb.dot(w))

