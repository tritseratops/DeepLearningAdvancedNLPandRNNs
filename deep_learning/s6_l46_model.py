import numpy as np

from s6_l46_generating_data import generate_data, plot_data
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
        #
        Z = np.tanh(X.dot(W) + b)
        Yp = Z.dot(V) + c
        # Yp = Yp.argmax(axis=1)
        Yp = Yp.reshape(-1, 1)
        return Z, Yp

    def get_dJdVm(self, T, Y, Z):
        return Z.T.dot(Y-T)

    def get_dJdc(self, T, Y):
        return (Y-T).sum()

    def get_dJdWdm(self, T, Y, V, Z, X):
        # T = T.reshape(-1, 1)
        return X.T.dot((Y-T).dot(V.T)*(1-Z*Z))

    def get_dJdBm(self, T, Y, V, Z, X):
        return ((Y-T).dot(V.T)*(1-Z*Z)).sum(axis=0)

    def gradient_step(self, X, T, Y, Z, W, b, V, c, learning_rate, reg_1):
        # derivative tanh: dtanh = 1-Y^2
        # newW = W + (1-np.power(np.tanh(X.dot(W) + b), 2))*learning_rate
        # T = T.reshape(-1, 1)
        newW = W - learning_rate*(self.get_dJdWdm(T, Y, V, Z, X)-reg_1*W)
        # newb = b + (1-np.power(np.tanh(X.dot(W) + b), 2)).sum(axis=1)*learning_rate
        newB = b - learning_rate*(self.get_dJdBm(T, Y, V, Z, X)-reg_1*b)
        # Z = np.tanh(X.dot(newW) + newb)
        # newV = V + np.exp(Z.dot(V) + c)*learning_rate
        newV = V - learning_rate*(self.get_dJdVm(T, Y, Z)-reg_1*V)
        # newc = V + np.exp(Z.dot(V) + c).sum(axis=1)*learning_rate
        newC = c - learning_rate*(self.get_dJdc(T, Y)-reg_1*c)
        return newW, newB, newV, newC

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, W, b, V, c, epochs=100, learning_rate=1e-5, reg1=0):
        for i in range(epochs):
            Z, Yp = self.predict(Xtrain, W, b, V, c)
            W, b, V, c = self.gradient_step(Xtrain, Ytrain, Yp, Z, W, b, V, c, learning_rate, reg1)
            # error = cross_entropy_error(Ytrain, Yp)
            error = mse(Ytrain, Yp)
            # Wsum = W.sum()
            # bsum = b.sum()
            # Vsum = V.sum()
            # csum = c.sum()
            # print("Wsum: ", Wsum)
            # print("bsum: ", bsum)
            # print("Vsum: ", Vsum)
            # print("csum: ", csum)
            # Ytrainsum = Ytrain.sum()
            # Ypsum = Yp.sum()
            # print("Ytrainsum: ", Ytrainsum)
            # print("Ypsum: ", Ypsum)
            print("i: ", i, ", mse: ", error)
        return W, b, V, c



def cross_entropy_error(T, Y):
    return np.mean((1-T)*np.log(1-Y)+T*np.exp(Y))


def stardard_error(T, Y):
    return (T-Y).mean()

def mse(T, Y): # mean squared error
    # return (np.power(T-Y,2)).mean()
    return (np.square(T - Y)).mean()




def main():
    X, Y = generate_data()
    X, Y = shuffle(X, Y)
    Y = Y.reshape(-1, 1)
    Xtrain = X[:-100, :]
    Ytrain = Y[:-100]
    Xtest = X[-100:, :]
    Ytest = Y[-100:]

    N = X.shape[0]
    D = X.shape[1]
    M = X.shape[1]+2
    K = 1
    model = deep_model(D, M, K)
    epochs = 100000
    learning_rate = 10e-5
    reg1 = 0.2



    model.W, model.b, model.V, model.c = model.fit(Xtrain, Ytrain, Xtest, Ytest, model.W, model.b, model.V, model.c, epochs, learning_rate, reg1)

    _, YpTrain = model.predict(Xtrain, model.W, model.b, model.V, model.c)
    # YpTrain = YpTrain.reshape(-1, 1)
    _, YpTest = model.predict(Xtest, model.W, model.b, model.V, model.c)
    # YpTest = YpTest.reshape(-1, 1)

    # print("Target:\n", Y)
    # print("Prediction:\n", Yp)
    print("Train MSE: ", mse(Ytrain, YpTrain))
    print("Test MSE: ", mse(Ytest, YpTest))

    # plot NN data
    plot_data(Xtrain, YpTrain, "Train prediction", 'g')

main()