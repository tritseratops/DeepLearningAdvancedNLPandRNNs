import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from s4_l22_load_data import load_data

class ecom_reviews_logreg_model():
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
        return 1/(1-np.exp(-a))

    def softmax(self, a):
        expa = np.exp(a)
        return expa/expa.sum(axis=0, keepdims=True)

    def predict(self, X, W, b):
        a = X.dot(W)+b
        return self.softmax(a)

    def gradient_step(self, X, T, Yp, learning_rate, W, b):
        # Yp = self.predict(X, W, b)
        W -= learning_rate*(X.T.dot(T-Yp))
        b -= learning_rate*((T-Yp).sum(axis=0))

        return W, b
    def classification_rate(self, T, Y):
        return np.mean(T==Y)

    def cross_entropy(self, T, Y):
        return (T*np.log(Y)).sum()

    def fit(self, X, T, learning_rate, EPOCHS):

        cl_rate_log = []
        ce_error_log = []
        for i in range(EPOCHS):
            Yp = self.predict(X, self.W, self.b)

            self.W, self.b = self.gradient_step(X, T, Yp, learning_rate, self.W, self.b)

            if i%100==0:
                ce = self.cross_entropy(T, Yp)
                cl_rate = self.classification_rate(T, Yp)
                cl_rate_log.append(cl_rate)
                ce_error_log.append(ce)

                print("i:", i, "ce:", ce, " cr:", cl_rate)
        return cl_rate_log, ce_error_log

def main():
    X, T = load_data()

    N = X.shape[0]
    D = X.shape[1]
    K = T.argmax()

    model = ecom_reviews_logreg_model(D, K)
    T2 = np.zeros((N, K))
    # hot-encode T
    for i in range(N):
        T2[i, T[i]]=1

    X, T = shuffle(X, T2)

    Xtrain = X[:-100, :]
    Ytrain = T[:-100, :]
    Xtest = X[-100:, :]
    Ytest = T[-100:, :]

    EPOCHS = 10000
    learning_rate = 10e-3


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

main()