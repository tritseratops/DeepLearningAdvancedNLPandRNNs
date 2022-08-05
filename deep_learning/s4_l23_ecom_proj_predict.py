import numpy as np

from s4_l22_load_data import load_data
from sklearn.utils import shuffle

def sigmoid(a):
    return 1/(1+np.exp(-a))

class ECom_model():
    """
    M - number of hidden layer nodes
    K - output number
    W - initial weights DxM
    b - initial bias  Vx1
    V - initial hidden weights MxK
    b - initial output bias Kx1
    """
    def __init__(self, W, b, V, c):
        self.W = W
        self.b = b
        self.V = V
        self.c = c

    """
    X -  input 
    T - target
    epochs - epochs to train
    learning_rate - learning rate
    M - number of hidden layer nodes
    K - output number
    """
    def fit(self, X, T, epochs, learning_rate):
        pass


    def softmax(self, X, W, b, V, c):
        # X NxD
        # W DxM
        # Z = sigmoid(X.dot(self.W)+self.b.T)
        Z = np.tanh(X.dot(W) + b)
        # Z NxM
        Y = np.exp(Z.dot(V)+ c)
        Y = Y/Y.sum(axis=1, keepdims=True)
        return Y

    def predict(self, X):
        Yp  = self.softmax(X, self.W, self.b, self.V, self.c)
        return Yp.argmax(axis=1)

def init_weights(D, M, K):
    W = sigmoid(np.random.randn(D, M))
    b = sigmoid(np.random.randn(M))

    V = sigmoid(np.random.randn(M, K))
    c = sigmoid(np.random.randn(K))

    return W, b, V, c

def classification_rate(T, Y):
    return np.mean(T==Y)

def main():
    X, T = load_data()

    X, T = shuffle(X, T)

    X_train = X[:-100, :]
    X_test = X[-100:, :]
    T_train = T[:-100]
    T_test = T[-100:]


    N = X_train.shape[0]
    D = X_train.shape[1]

    M = D + 1
    K = int(T_train.max()+1) # max categories
    # init random weights
    W, b, V, c = init_weights(D, M, K)

    epochs = 10000
    learning_rate = 1
    model = ECom_model(W, b, V, c)

    Yp_test = model.predict(X_test)

    print("Classification rate on random:", classification_rate(T_test, Yp_test))

main()