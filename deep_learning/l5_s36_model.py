import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def generate_data():
    N = 500
    D = 2

    X1 = np.random.randn(N, D) + np.array([4, 4])
    X2 = np.random.randn(N, D) + np.array([4,0])
    X3 = np.random.randn(N, D) + np.array([0, 4])

    X = np.concatenate((X1, X2, X3))
    Y = np.zeros((N+N+N, 3))
    Y[N:2*N,1]=1
    Y[2*N:,2]=2
    c = Y.argmax(axis=1)
    plt.scatter(X[:,0], X[:, 1], c=c)
    plt.show()

    return X, Y


# generate_data()

class cat_3_circles_model():
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
        if V is None:
            self.V = self.sigmoid(np.random.randn(M, K))
            self.c = self.sigmoid(np.random.randn(K))
        else:
            self.V = V
            self.c = c

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, a):
        expa= np.exp(a)
        expasum =expa.sum(axis=1)
        expa = np.exp(a)
        # print(expasum)
        return (expa/expa.sum(axis=1).reshape(expa.shape[0],1))

    def predict(self, X, W, b, V, c):
        z = self.sigmoid(X.dot(W)+b)
        Yp = self.softmax(z.dot(V)+c)
        return Yp, z

    def cross_entropy(self, Yp, T):
        return ((1-T).T.dot(np.log(1-Yp))+T.T.dot(Yp)).mean()

    def cost(self, T, Y):
        c = T*np.log(Y)
        return c.sum()

    def classification_rate(self, T, Y):
        Targmax = T.argmax(axis=1)
        Yargmax = Y.argmax(axis=1)
        c = (np.count_nonzero(Targmax==Yargmax))/Targmax.shape[0]
        return c


    def save(self, filename='data/s5l36_model.csv'):
        model = self.W
        # model.append(self.b)
        model = np.append(model, self.b)
        model = np.append(model, self.V)
        model = np.append(model, self.c)

        np.savetxt(filename, model, delimiter=",")

    def load(self, filename='data/s5l36_model.csv'):
        model = np.loadtxt(filename, delimiter=',')
        self.W = model[:-1]
        self.b = model[-1]

    def gradient_step(self, learning_rate, X, T, W, b, V, c):
        Yp, Z = self.predict(X, W, b, V, c)
        dJdWmd = ((T-Yp).dot(V.T).dot(Z.T).dot(Z-1).T.dot(X)).sum()
        dJdb = ((T-Yp).dot(V.T).dot(Z.T).dot(Z-1)).sum()
        dJdVmk = Z.T.dot(T-Yp).sum()
        dJdCk = (T-Yp).sum()
        W = W + learning_rate*dJdWmd
        b = b + learning_rate*dJdb
        V = V + learning_rate*dJdVmk
        c = c + learning_rate*dJdCk
        return W, b, V, c

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate, epochs):
        train_costs = []
        test_costs = []
        for i in range(epochs):
            # get next state
            self.W, self.b, self.V, self.c = self.gradient_step(learning_rate, Xtrain, Ytrain, self.W, self.b, self.V, self.c)

            Yp_train, Z  = self.predict(Xtrain, self.W, self.b, self.V, self.c)
            Yp_test, _ = self.predict(Xtest, self.W, self.b, self.V, self.c)

            if i%10==0:
                # calculate new cost
                train_cr= self.classification_rate(Ytrain, Yp_train)
                train_cost = self.cost(Ytrain, Yp_train)

                # add cost to log
                train_costs.append(train_cost)
                print("i: ", i, " train classification rate:", train_cr)
                print("i: ", i, " train cost:", train_cost)

                # calculate new cost
                test_cr = self.classification_rate(Ytest, Yp_test)
                test_cost = self.cost(Ytest, Yp_test)

                # add cost to log
                test_costs.append(test_cost)
                print("i: ", i, " test classification rate:", test_cr)
                print("i: ", i, " test cost:", test_cost)

        return    train_costs, test_costs

def main():
    X, Y = generate_data()
    X, Y = shuffle(X, Y)
    Xtrain = X[:-100, :]
    Ytrain = Y[:-100, :]
    Xtest = X[-100:, :]
    Ytest = Y[-100:, :]

    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    M = 10 # user selected, inner dimention can be changed
    K = 3 # we know we have 3 categories
    learning_rate = 1E-4
    EPOCHS  = 10000


    model = cat_3_circles_model(D=D, M=M, K=K)
    train_log, test_log  = model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate, EPOCHS)

    # plot costs
    plt.plot(train_log)
    plt.title("Cross entropy train")
    plt.plot(test_log)
    plt.title("Cross entropy test")
    plt.legend()
    plt.show()


main()


