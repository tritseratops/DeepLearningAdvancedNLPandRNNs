import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    N = 500
    D = 2

    X1 = np.random.randn(N, D) + np.array([4, 4])
    X2 = np.random.randn(N, D) + np.array([4,0])
    X3 = np.random.randn(N, D) + np.array([0, 4])

    X = np.concatenate((X1, X2, X3))
    Y = np.zeros(N+N+N)
    Y[N:2*N]=1
    Y[2*N:]=2
    plt.scatter(X[:,0], X[:, 1], c=Y)
    plt.show()


# generate_data()

class model():
    def __init__(self, N, D, M, K, W, b, V, c):
        self.N = N
        self.D = D
        self.M = M
        self.K = K
        self.W = W
        self.b = b
        self.V = V
        self.c = c

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, a):
        expa= np.exp(a)
        return (expa/expa.sum(axis=1))

    def predict(self, X, W, b, V, c):
        z = self.sigmoid(X.dot(W)+b)
        Yp = self.softmax(z.dot(V)+c)
        return Yp

    def cross_entropy(self, Yp, T):
        return ((1-T).dot(np.log(1-Yp))+T.dot(Yp)).mean()

    def gradient_step(self, learning_rate, X, T, Z, W, b, V, c):
        Yp = self.predict()
        dJdWmd = ((T-Yp).dot(V).dot(Z).dot(Z-1).dot(X)).sum()
        dJdb = ((T-Yp).dot(V).dot(Z).dot(Z-1)).sum()
        dJdVmk = Z.dot(T-Yp).sum()
        dJdCk = (T-Yp).sum()
        W = W - learning_rate*dJdWmd
        b = b - learning_rate*dJdb
        V = V - learning_rate*dJdVmk
        c = c - learning_rate*dJdCk
        return W, b, V, c



