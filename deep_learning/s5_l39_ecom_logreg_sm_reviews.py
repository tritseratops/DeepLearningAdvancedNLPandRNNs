import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from s4_l22_load_data import load_data


def get_binary_data():
    # return only the data from the first 2 classes
    # Xtrain, Ytrain, Xtest, Ytest = load_data()
    # X2train = Xtrain[Ytrain <= 1]
    # Y2train = Ytrain[Ytrain <= 1]
    # X2test = Xtest[Ytest <= 1]
    # Y2test = Ytest[Ytest <= 1]

    X, Y = load_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


class EcomReviewsLogRegModel():
    def __init__(self, D=None, K=None, W=None, b=None):
        if D is None:
            return
        if K is None:
            return
        if W is None:
            self.W = self.sigmoid(np.random.randn(D, K))
            self.b = self.sigmoid(np.random.randn(K))
        else:
            self.W = W
            self.b = b

    def sigmoid(self, a):
        return 1 / (1 - np.exp(-a))

    def softmax(self, a):
        expa = np.exp(a)
        return expa / expa.sum(axis=1, keepdims=True)

    def predict(self, X, W, b):
        a = X.dot(W) + b
        return self.softmax(a)

    def gradient_step(self, X, T, Yp, learning_rate, W, b):
        # Yp = self.predict(X, W, b)
        W -= learning_rate * X.T.dot(Yp - T)
        b -= learning_rate * (Yp - T).sum(axis=0)

        # check if nan is in W - output X, T, Yp
        if np.isnan(W.sum()):
            print("X:", X)
            print("T:", T)
            print("Yp:", Yp)
            print("W:", W, " b:", b)
            breakpoint()
        return W, b

    def classification_rate(self, T, Y):
        return np.mean(T == Y)

    def cross_entropy(self, T, Y):
        return -np.mean(T * np.log(Y))

    def fit(self, X, T, learning_rate, epochs):

        cl_rate_log = []
        ce_error_log = []
        for i in range(epochs):
            Yp = self.predict(X, self.W, self.b)

            self.W, self.b = self.gradient_step(X, T, Yp, learning_rate, self.W, self.b)

            if i % 100 == 0:
                ce = self.cross_entropy(T, Yp)
                cl_rate = self.classification_rate(T, Yp)
                cl_rate_log.append(cl_rate)
                ce_error_log.append(ce)

                print("i:", i, "ce:", ce, " cr:", cl_rate)
                # print("W:", self.W, " b:", self.b)
        return cl_rate_log, ce_error_log


def main():
    X, T = get_binary_data()

    N = X.shape[0]
    D = X.shape[1]
    K = T.max() + 1

    model = EcomReviewsLogRegModel(D, K)
    T2 = np.zeros((N, K))
    # hot-encode T
    for i in range(N):
        T2[i, T[i]] = 1

    X, T = shuffle(X, T2)

    Xtrain = X[:-100, :]
    Ytrain = T[:-100, :]
    Xtest = X[-100:, :]
    Ytest = T[-100:, :]

    EPOCHS = 10000
    learning_rate = 10e-4

    cr_log, ce_log = model.fit(Xtrain, Ytrain, learning_rate, EPOCHS)

    plt.plot(cr_log)
    plt.title("Classification Rate")
    plt.legend()
    plt.show()

    plt.plot(ce_log)
    plt.title("Cross Entropy")
    plt.legend()
    plt.show()

    Yptest = model.predict(Xtest, model.W, model.b)
    test_ce = model.cross_entropy(Ytest, Yptest)
    test_cr = model.classification_rate(Ytest, Yptest)
    print("Train ce:", ce_log[-1], " cr:", cr_log[-1])
    print("Test ce:", test_ce, " cr:", test_cr)
    print("W:", model.W, " b:", model.b)


main()
