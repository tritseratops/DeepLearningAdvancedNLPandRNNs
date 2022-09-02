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
    Y = np.zeros(N+N+N)
    Y[N:2*N]=1
    Y[2*N:]=2
    plt.scatter(X[:,0], X[:, 1], c=Y)
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
        return (expa/expa.sum(axis=1))

    def predict(self, X, W, b, V, c):
        z = self.sigmoid(X.dot(W)+b)
        Yp = self.softmax(z.dot(V)+c)
        return Yp, z

    def cross_entropy(self, Yp, T):
        return ((1-T).dot(np.log(1-Yp))+T.dot(Yp)).mean()

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
        dJdWmd = ((T-Yp).dot(V).dot(Z).dot(Z-1).dot(X)).sum()
        dJdb = ((T-Yp).dot(V).dot(Z).dot(Z-1)).sum()
        dJdVmk = Z.dot(T-Yp).sum()
        dJdCk = (T-Yp).sum()
        W = W - learning_rate*dJdWmd
        b = b - learning_rate*dJdb
        V = V - learning_rate*dJdVmk
        c = c - learning_rate*dJdCk
        return W, b, V, c

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate, epochs):
        train_costs = []
        test_costs = []
        for i in range(epochs):
            # get next state
            self.W, self.b, self.V, self.c = self.gradient_step(learning_rate, Xtrain, Ytrain, self.W, self.b, self.V, self.c)

            Yp, Z  = self.predict(Xtrain, self.W, self.b, self.V, self.c)

            if i%10==0:
                # calculate new cost
                train_cost= self.cross_entropy(Yp, Ytrain)

                # add cost to log
                train_costs.append(i, train_cost)
                print("Train cost:", train_cost)

                # calculate new cost
                test_cost = self.cross_entropy(Yp, Ytrain)

                # add cost to log
                test_costs.append(i, test_cost)
                print("Test cost:", test_cost)

        return    train_costs, test_costs

def main():
    X, Y = generate_data()
    X, Y = shuffle(X, Y)
    Xtrain = X[:-100, :]
    Ytrain = X[:-100, :]
    Xtest = X[-100:, :]
    Ytest = X[-100:, :]

    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    M = 5 # user selected, inner dimention can be changed
    K = 3 # we know we have 3 categories
    learning_rate = 1E-6
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


