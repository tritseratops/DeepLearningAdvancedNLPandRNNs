import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_xor_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0])
    return X, Y


class NNTanhSigmaModel():
    def __init__(self, D=None, M=None, K=None, W=None, b=None, V=None, c=None):
        if D is None:
            return
        if M is None:
            return
        if K is None:
            return
        if W is None:
            self.W = self.sigmoid(np.random.randn(D, M))
            self.b = self.sigmoid(np.random.randn(M))
        else:
            self.W = W
            self.b = b
        if W is None:
            self.V = self.sigmoid(np.random.randn(M, K))
            self.c = self.sigmoid(np.random.randn(K))
        else:
            self.V = V
            self.c = c

    def sigmoid(self, a):
        return 1 / (1 - np.exp(-a))

    def softmax(self, a):
        expa = np.exp(a)
        return expa / expa.sum(axis=1, keepdims=True)

    def predict(self, X, W, b, V, c):
        Z = X.dot(W) + b
        Z = np.tanh(Z)
        a = Z.dot(V) + c
        return Z, self.softmax(a)

    def get_dJdWdm(self, X, Y, T, V, Z):
        dJdWdm = X.T.dot((T-Y).dot(V.T)*(1-np.power(Z,2)))
        return dJdWdm

    def get_dJdBk(self, Y, T, V, Z):
        dJdBk = ((T-Y).dot(V.T)*(1-np.power(Z,2))).sum(axis=0)
        return dJdBk

    def get_dJdVmk(self, T, Y, Z):
        JdVmk = Z.T.dot(T-Y)
        return JdVmk

    def get_dJdCk(self,T, Y):
        JdCk = (T - Y).sum(axis=0)
        return JdCk

    def gradient_step(self, X, T, Yp, learning_rate, regularization, Z, W, b, V, c):
        # Yp = self.predict(X, W, b, V,c)
        W += learning_rate * (self.get_dJdWdm(X, Yp, T, V, Z) -  regularization*W)
        b += learning_rate * (self.get_dJdBk(Yp, T, V, Z) -  regularization*b)
        V += learning_rate * (self.get_dJdVmk(T, Yp, Z) -  regularization*V)
        c += learning_rate * (self.get_dJdCk(T, Yp) -  regularization*c)

        # check if nan is in W - output X, T, Yp
        if np.isnan(W.sum()):
            print("X:", X)
            print("T:", T)
            print("Yp:", Yp)
            print("W:", W, " b:", b)
            breakpoint()
        return W, b, V, c

    def classification_rate(self, T, Y):
        Targmax = np.argmax(T, axis=1)
        Yargmax = np.argmax(Y, axis=1)

        return np.mean(Targmax == Yargmax)

    def cross_entropy(self, T, Y):
        return -np.mean(T * np.log(Y))

    def fit(self, X, T, Xtest, Ttest, learning_rate, regularization, epochs):

        cl_rate_log = []
        ce_error_log = []
        cl_test_log = []
        for i in range(epochs):
            Z, Yp = self.predict(X, self.W, self.b, self.V, self.c)
            self.W, self.b, self.V, self.c = self.gradient_step(X, T, Yp, learning_rate, regularization, Z, self.W, self.b, self.V, self.c)

            if i % 100 == 0:
                ce = self.cross_entropy(T, Yp)
                cl_rate = self.classification_rate(T, Yp)
                cl_rate_log.append(cl_rate)
                ce_error_log.append(ce)

                Ztest, Yptest = self.predict(Xtest, self.W, self.b, self.V, self.c)
                cl_rate = self.classification_rate(Ttest, Yptest)
                cl_test_log.append(cl_rate)

                print("i:", i, "ce:", ce, " cr:", cl_rate)
                # print("W:", self.W, " b:", self.b)
        return cl_rate_log, ce_error_log, cl_test_log


def main():
    # X, T = get_binary_data()
    X, T = load_xor_data()
    # load donut data
    N = X.shape[0]
    D = X.shape[1]
    M = 3
    K = T.max() + 1

    model = NNTanhSigmaModel(D, M, K)
    T2 = np.zeros((N, K))
    # hot-encode T
    for i in range(N):
        T2[i, T[i]] = 1

    X, T = shuffle(X, T2)

    Xtrain = X[:-100, :]
    Ytrain = T[:-100, :]
    Xtest = X[-100:, :]
    Ytest = T[-100:, :]

    EPOCHS = 100000
    learning_rate = 10e-4
    regularization = 0.2

    cr_log, ce_log, cr_test_log = model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate, regularization, EPOCHS)

    X = np.arange(len(cr_log))
    plt.plot(X, cr_log, color='b', label='Train Classification Rate')
    plt.plot(X, cr_test_log, color='g', label='Test Classification Rate')
    # plt.plot(cr_log)
    # plt.title("Classification Rate")
    # plt.plot(cr_test_log)
    # plt.title("Classification Rate")
    plt.legend()
    plt.show()

    plt.plot(ce_log)
    plt.title("Cross Entropy")
    plt.legend()
    plt.show()

    _, Yptest = model.predict(Xtest, model.W, model.b, model.V, model.c)
    test_ce = model.cross_entropy(Ytest, Yptest)
    test_cr = model.classification_rate(Ytest, Yptest)
    print("Train ce:", ce_log[-1], " cr:", cr_log[-1])
    print("Test ce:", test_ce, " cr:", test_cr)
    print("W:", model.W, " b:", model.b)


main()
