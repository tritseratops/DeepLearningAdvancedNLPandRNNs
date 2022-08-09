import numpy as np

from s6_l46_generating_data import generate_data
from sklearn.utils import shuffle

class deep_model():
    # N- items, D - features, M, hidden dimations, K - output dimentions
    def __init__(self, D, M, K):
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

    def forward(self, X, T, W, b, V, c, learning_rate):
        # derivative tanh: dtanh = 1-Y^2
        newW = W + (1-np.power(np.tanh(X.dot(W) + b), 2))*learning_rate
        newb = b + (1-np.power(np.tanh(X.dot(W) + b), 2)).sum(axis=1)*learning_rate
        Z = np.tanh(X.dot(newW) + newb)
        newV = V + np.exp(Z.dot(V) + c)*learning_rate
        newc = V + np.exp(Z.dot(V) + c).sum(axis=1)*learning_rate
        return newW, newb, newV, newc

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, W, b, V, c, epochs=100, learning_rate=1e-5):
        for i in range(epochs):
            W, b, V, c = self.forward(Xtrain, Ytrain, W, b, V, c, learning_rate)
            Yp = self.predict(Xtest, W, b, V, c)
            error = cross_entropy_error(Ytest, Yp)
            print(error)
        return W, b, V, c



def cross_entropy_error(T, Y):
    return np.mean((1-T)*np.log(1-Y)+T*np.exp(Y))

def main():
    X, Y = generate_data()
    X, Y = shuffle(X, Y)
    Xtrain = X[:-100, :]
    Ytrain = Y[:-100]
    Xtest = X[-100:, :]
    Ytest = Y[-100:]

    N = X.shape[0]
    D = X.shape[1]
    M = X.shape[1]+2
    K = 1
    model = deep_model(D, M, K)
    epochs = 10000

    Yp = model.predict(X, model.W, model.b, model.V, model.c)

    W, b, V, c = model.fit(Xtrain, Ytrain, Xtest, Ytest, model.W, model.b, model.V, model.c)

    print("Target:\n", Y)
    print("Prediction:\n", Yp)
    print(cross_entropy_error(Y, Yp))


main()