import numpy as np

from s6_l46_generating_data import generate_data

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
        return Yp.mean()/Yp.sum(axis=1)


    def predict(self, X, W, b, V, c):
        Z = np.tanh(X.dot(W) + b)
        Yp = np.exp(Z.dot(V) + c)
        return Yp


def cross_entropy_error(T, Y):
    return np.mean((1-T)*np.log(1-Y)+T*np.exp(Y))

def main():
    X, Y = generate_data()
    N = X.shape[0]
    D = X.shape[1]
    M = X.shape[1]+2
    K = 1
    model = deep_model(N, D, M, K)

    Yp = model.predict(X, model.W, model.b, model.V, model.c)

    print("Target:\n", Y)
    print("Prediction:\n", Yp)
    print(cross_entropy_error(Y, Yp))


main()