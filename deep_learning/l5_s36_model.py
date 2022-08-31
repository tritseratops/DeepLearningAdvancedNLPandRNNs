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


generate_data()