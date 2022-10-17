import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from s8_l64_facial_data import get_data, get_emotion, get_emotion_if
from datetime import datetime
import time
import json
class LogisticSoftmaxModel():
    def __init__(self, D=None, K=None, W=None, b=None, V=None, c=None):
        if D is None:
            return
        if K is None:
            return
        if W is None:
            # self.W = self.sigmoid(np.random.randn(D, K))
            # self.b = self.sigmoid(np.random.randn(K))
            self.W = np.random.randn(D, K)/np.sqrt(D+K)
            self.b = np.random.randn(K)/np.sqrt(K)
        else:
            self.W = W
            self.b = b

    def sigmoid(self, a):
        return 1 / (1 - np.exp(-a))

    def softmax(self, a):
        expa = np.exp(a)
        return expa / expa.sum(axis=1, keepdims=True)

    def relu(X):
        return np.maximum(0, X)

    def predict(self, X, W, b):
        a = X.dot(W) + b
        # tahn
        # Z = np.tanh(Z)
        # sigmoid
        # Z = self.sigmoid(Z)
        # relU
        # Z = self.relu(Z)
        # a = Z.dot(V) + c
        return self.softmax(a)

    def get_dJdWdm(self, X, Y, T):
        dJdWdm = X.T.dot(T-Y)
        return dJdWdm

    def get_dJdBk(self, Y, T):
        # tahn
        dJdBk = (T-Y).sum(axis=0)
        # sigmoid
        # dJdBk = ((T - Y).dot(V.T) * (Z*(1 - Z))).sum(axis=0)
        # relU
        # dJdBk = ((T - Y).dot(V.T)).sum(axis=0)
        return dJdBk


    def gradient_step(self, X, T, Yp, learning_rate, W, b, regularization1=0, regularization2=0):
        # Yp = self.predict(X, W, b, V,c)
        W += learning_rate * (self.get_dJdWdm(X, Yp, T)+regularization1*W+regularization2*np.square(W))
        b += learning_rate * (self.get_dJdBk(Yp, T)+regularization1*b+regularization2*np.square(b))

        # check if nan is in W - output X, T, Yp
        if np.isnan(W.sum()):
            print("X:", X)
            print("T:", T)
            print("Yp:", Yp)
            print("W:", W, " b:", b)
            breakpoint()
        return W, b

    def classification_rate(self, T, Y):
        Targmax = np.argmax(T, axis=1)
        Yargmax = np.argmax(Y, axis=1)

        return np.mean(Targmax == Yargmax)

    def cross_entropy(self, T, Y):
        return -np.mean(T * np.log(Y))

    def fit(self, X, T, Xtest, Ttest, learning_rate, epochs, regularization1=0, regularization2=0):

        cl_rate_log = []
        ce_error_log = []
        cl_test_log = []
        for i in range(epochs):
            Yp = self.predict(X, self.W, self.b)
            self.W, self.b = self.gradient_step(X, T, Yp, learning_rate, self.W, self.b, regularization1, regularization2)

            if i % 1 == 0:
                ce = self.cross_entropy(T, Yp)
                cl_rate = self.classification_rate(T, Yp)
                cl_rate_log.append(cl_rate)
                ce_error_log.append(ce)

                Yptest = self.predict(Xtest, self.W, self.b)
                cl_rate = self.classification_rate(Ttest, Yptest)
                cl_test_log.append(cl_rate)

                print("i:", i, "ce:", ce, " cr:", cl_rate)
                # print("W:", self.W, " b:", self.b)
        return cl_rate_log, ce_error_log, cl_test_log
    def save(self, filename='face_model.csv'):
        # model = self.W
        # # model.append(self.b)
        # model = np.append(model, self.b)
        # model = np.append(model, self.V)
        # model = np.append(model, self.c)
        # np.savetxt(filename, model, delimiter=",")
        with open(filename, 'w') as fp:
            listW = self.W.tolist()
            listb = self.b.tolist()
            json.dump({'W': listW, 'b': listb}, fp)

    def load(self, filename='face_model.csv'):
        # model = np.loadtxt(filename,  delimiter=',')
        # self.W = model[:-1]
        # self.b = model[-1]

        with open(filename, 'r') as fp:
            model_weights = json.load(fp)
            self.W = np.array(model_weights['W'])
            self.b = np.array(model_weights['b'])



def predict(model):
    X, Y = get_data()
    # 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1  =np.repeat(X1, 9, axis = 0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    # show face and predict
    # get random face
    Py  = model.predict(X)
    continue_y = True
    while (continue_y):
        face_id = np.random.randint(X.shape[0])
        face_flat = X[face_id, :]
        face = np.reshape(face_flat, (48,48))
        plt.imshow(face)
        plt.show()
        print("Prediction:",get_emotion(Py[face_id]))
        start = time.time()
        emotion = get_emotion(Y[face_id])
        end = time.time()
        print("Result:", emotion)
        print("Emotion time match:", end - start)
        start = time.time()
        emotion = get_emotion(Y[face_id])
        end = time.time()
        print("Emotion time if:", end - start)
        print("Result:", emotion)
        y = input("Do you want to continue? (y/n) ")
        if y.lower()!='y':
            continue_y = False
    exit()



def main():
    X, T = get_data()  # for all outputs
    N = X.shape[0]
    D = X.shape[1]
    M = 10
    K = T.max() + 1

    model = LogisticSoftmaxModel(D, K)
    T2 = np.zeros((N, K))

    # hot-encode T
    for i in range(N):
        T2[i, T[i]] = 1

    X, T = shuffle(X, T2)

    Xtrain = X[:-100, :]
    Ytrain = T[:-100, :]
    Xtest = X[-100:, :]
    Ytest = T[-100:, :]

    EPOCHS = 1000
    learning_rate = 10e-8
    reg1 = 0.1
    reg2 = 0.1
    model.load()
    cr_log, ce_log, cr_test_log = model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate, EPOCHS, reg1,reg2)

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

    Yptest = model.predict(Xtest, model.W, model.b)
    test_ce = model.cross_entropy(Ytest, Yptest)
    test_cr = model.classification_rate(Ytest, Yptest)
    print("Train ce:", ce_log[-1], " cr:", cr_log[-1])
    print("Test ce:", test_ce, " cr:", test_cr)
    print("W:", model.W, " b:", model.b)



    model.save()
    # predict(model)




if __name__ == "__main__":
    main()