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
    Y[0:N, 0] = 1
    Y[N:2*N,1] =1
    Y[2*N:,2] =1
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
        # expa= np.exp(a)
        # expasum =expa.sum(axis=1)
        expa = np.exp(a)
        # print(expasum)
        return expa/expa.sum(axis=1).reshape(expa.shape[0],1)

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

    def get_dJdWdm(self, T, Yp, V, Z, X):
        dJdWdm = ((T-Yp).dot(V.T).dot(Z.T).dot(1-Z).T.dot(X)).T
        N = T.shape[0]
        D = X.shape[1]
        M = V.shape[0]
        K = T.shape[1]
        # dJdWdm = np.ndarray((D,M))
        # for d in range(D):
        #     for m in range(M):
        #         for n in range(N):
        #             for k in range(K):
        #                 dJdWdm[d,m] +=(T[n,k]-Yp[n,k])*V[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]

        return dJdWdm

    def get_dJdBm(self, T, Yp, V, Z):
        dJdBm = ((T-Yp).dot(V.T).dot(Z.T).dot(1-Z)).sum(axis=0)
        # N = T.shape[0]
        # M = V.shape[0]
        # K = T.shape[1]
        # dJdBm = np.ndarray((M))
        # for m in range(M):
        #     for n in range(N):
        #         for k in range(K):
        #             dJdBm[m] +=(T[n,k]-Yp[n,k])*V[m,k]*Z[n,m]*(1-Z[n,m])

        return dJdBm

    def get_dJdVmk(self, T, Yp, Z):
        dJdVmk = Z.T.dot(T-Yp)
        # N = T.shape[0]
        # M = Z.shape[1]
        # K = T.shape[1]
        # # dJdVmk = np.ndarray((M, K))
        # for m in range(M):
        #     for k in range(K):
        #         for n in range(N):
        #             dJdVmk[m,k] +=Z[n,m]*(T[n,k]-Yp[n,k])

        return dJdVmk

    def get_dJdCk(self, T, Yp):
        dJdCk = (T-Yp).sum(axis=0)
        # N = T.shape[0]
        # K = T.shape[1]
        # dJdCk = np.ndarray((K))
        # for k in range(K):
        #     for n in range(N):
        #         dJdCk[k] +=(T[n,k]-Yp[n,k])

        return dJdCk

    def gradient_step(self, learning_rate, X, T, W, b, V, c):
        Yp, Z = self.predict(X, W, b, V, c)
        # dJdWmd = ((T-Yp).dot(V.T).dot(Z.T).dot(1-Z).T.dot(X)).T
        dJdWdm = self.get_dJdWdm(T, Yp, V, Z, X)
        # dJdb = ((T-Yp).dot(V.T).dot(Z.T).dot(1-Z)).sum(axis=0)
        dJdb = self.get_dJdBm(T, Yp, V, Z)
        # dJdVmk = Z.T.dot(T-Yp)
        dJdVmk = self.get_dJdVmk(T, Yp, Z)
        # dJdCk = (T-Yp).sum(axis=0)
        dJdCk = self.get_dJdCk(T, Yp)
        W = W + learning_rate*dJdWdm
        b = b + learning_rate*dJdb
        V = V + learning_rate*dJdVmk
        c = c + learning_rate*dJdCk
        return W, b, V, c

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate, epochs):
        train_costs = []
        test_costs = []
        train_cr_log = []
        for i in range(epochs):
            # get next state
            self.W, self.b, self.V, self.c = self.gradient_step(learning_rate, Xtrain, Ytrain, self.W, self.b, self.V, self.c)

            Yp_train, Z  = self.predict(Xtrain, self.W, self.b, self.V, self.c)
            #
            # Wtest, btest, Vtest, ctest = self.gradient_step(learning_rate, Xtest, Ytest, self.W, self.b, self.V,
            #                                                     self.c)
            #
            # Yp_test, _ = self.predict(Xtest, self.W, self.b, self.V, self.c)

            if i%100==0:
                # calculate new cost
                train_cr= self.classification_rate(Ytrain, Yp_train)
                train_cost = self.cost(Ytrain, Yp_train)

                # add cost to log
                train_costs.append(train_cost)
                train_cr_log.append(train_cr)
                print("i: ", i, " train classification rate:", train_cr)
                print("i: ", i, " train cost:", train_cost)
                print("W:", self.W, " b:", self.b, " V:", self.V, "c:", self.c)
                # # calculate new cost
                # test_cr = self.classification_rate(Ytest, Yp_test)
                # test_cost = self.cost(Ytest, Yp_test)
                #
                # # add cost to log
                # test_costs.append(test_cost)
                # print("i: ", i, " test classification rate:", test_cr)
                # print("i: ", i, " test cost:", test_cost)

        return    train_costs, train_cr_log, test_costs

def main():
    X, Y = generate_data()
    X, Y = shuffle(X, Y)
    Xtrain = X[:-100, :]
    Ytrain = Y[:-100, :]
    Xtest = X[-100:, :]
    Ytest = Y[-100:, :]

    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    M = 3 # user selected, inner dimension can be changed
    K = 3 # we know we have 3 categories
    learning_rate = 1e-3
    EPOCHS  = 1000


    model = cat_3_circles_model(D=D, M=M, K=K)
    train_log, train_cr, test_log  = model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate, EPOCHS)

    # plot costs
    plt.plot(train_log)
    plt.title("Cross entropy train")
    plt.legend()
    plt.show()
    plt.plot(train_cr)
    plt.title("Classification rate")
    # plt.plot(test_log)
    # plt.title("Cross entropy test")
    plt.legend()
    plt.show()


main()


