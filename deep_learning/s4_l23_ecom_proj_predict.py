import numpy as np

from s4_l22_load_data import load_data

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
    def __init__(self, M, K, W, b, V, c):
        self.M = M
        self.K = K
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

    def predict(self):
        pass

def main():
    X, Y = load_data()



main()