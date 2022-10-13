import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from s8_l64_facial_data import get_data, get_emotion, get_emotion_if
from datetime import datetime
import time
class NNTanhSoftmaxModel():
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

    def gradient_step(self, X, T, Yp, learning_rate, Z, W, b, V, c):
        # Yp = self.predict(X, W, b, V,c)
        W += learning_rate * self.get_dJdWdm(X, Yp, T, V, Z)
        b += learning_rate * self.get_dJdBk(Yp, T, V, Z)
        V += learning_rate * self.get_dJdVmk(T, Yp, Z)
        c += learning_rate * self.get_dJdCk(T, Yp)

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

    def fit(self, X, T, Xtest, Ttest, learning_rate, epochs):

        cl_rate_log = []
        ce_error_log = []
        cl_test_log = []
        for i in range(epochs):
            Z, Yp = self.predict(X, self.W, self.b, self.V, self.c)
            self.W, self.b, self.V, self.c = self.gradient_step(X, T, Yp, learning_rate, Z, self.W, self.b, self.V, self.c)

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
    def save(self, filename='face_model.csv'):
        model = self.W
        # model.append(self.b)
        model = np.append(model, self.b)
        np.savetxt(filename, model, delimiter=",")

    def load(self, filename='face_model.csv'):
        model = np.loadtxt(filename,  delimiter=',')
        self.W = model[:-1]
        self.b = model[-1]

def train(starting_learning_rate=5e-6, epochs=120000, starting_model=None):
    X, Y = get_data()

    # X0 = X[Y==0, :]
    # X1 = X[Y==1, :]
    # X1  = np.repeat(X1, 9, axis = 0)
    # X = np.vstack([X0, X1])
    # Y = np.array([0]*len(X0) + [1]*len(X1))

    print("Start training:", datetime.now())
    if not starting_model:
        model = NNTanhSoftmaxModel()
    else:
        model = starting_model
    model.fit(X, Y, starting_learning_rate=starting_learning_rate, epochs=epochs, show_fig=True)
    model.score(X, Y)
    print("End training:", datetime. now())
    print("W:", model.W)
    print("b:", model.b)

    return model


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

    model = NNTanhSoftmaxModel()
    # model.load()
    model = train(starting_learning_rate=1e-6, epochs=10000, starting_model=model)
    model.save()
    # predict(model)



if __name__ == "__main__":
    main()