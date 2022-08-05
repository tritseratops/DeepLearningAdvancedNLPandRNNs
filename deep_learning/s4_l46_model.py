import numpy as np

class deep_model():
    # N- items, D - features, M, hidden dimations, K - output dimentions
    def __init__(self, N, D, M, K):
        self.W = np.random.randn(D, M)
        self.b = np.random.randn(M)
        self.V = np.random.randn(M, K)
        self.c = np.random.randn(K)

    def softmax(self, X, W, b, V, c):
        Z = np.tanh(X.dot(W)+b)
        Yp =np.exp(Z.dot(V)+c)
        return Yp.mean()/Yp.Sum(axis=1)

    def predict(self, X, W, b, V, c):
        Yp = self.softmax(X, W, b, V, c)
        K = V.shape[1] # number of outputs
        Y = np.argmax(Yp)


def main():
    # X, Y = load_data()
    model = deep_model()
