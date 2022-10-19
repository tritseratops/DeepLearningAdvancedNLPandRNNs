import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from s6l40_utils import getBinaryData, sigmoid, sigmoid_cost, error_rate, get_emotion
from datetime import datetime
import time
class LogisticModel(object):
    def __init__(self):
        self.W = None
        self.b = None

    # reg - regularization penalty
    def fit(self, X, Y, starting_learning_rate = 1e-6, reg=0., epochs=120000, show_fig=False):
        X, Y = shuffle(X,Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape

        if self.W is None:
            self.W = np.random.randn(D)/np.sqrt(D)
            self.b = 0

        costs = []
        best_validation_error = 1
        learning_rate = starting_learning_rate
        for i in range(epochs):
            # forward propagation and cost calculation
            pY = self.forward(X)

            # gradient descent step
            self.W  -= learning_rate*(X.T.dot(pY-Y)+reg*self.W)
            self.b -= learning_rate * ((pY - Y).sum() + reg * self.b)

            # log cost
            decreasing_cost_count=0
            first_cost = True
            if i%20==0:
                pYvalid = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                # adaptive gradient descent
                # if cost increases we decrease learning rate x2
                if not first_cost and c>costs[-1]:
                    first_cost = False
                    learning_rate /= 2
                    print("Learning rate change to:", learning_rate)
                    decreasing_cost_count= 0
                else:
                    decreasing_cost_count += 1
                # if cost decreases for 10 times in sequence we increase learning rate
                if decreasing_cost_count>9:
                    learning_rate *=2
                    print("Learning rate change to:", learning_rate)

                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print("i: ", i, " cost:", c, " error:", e)
                if e<best_validation_error:
                    best_validation_error = e
        print("Best validation error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.W)+self.b)

    def predict(self, X):
        Py = self.forward(X)
        return np.round(Py)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1-error_rate(Y, prediction)

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
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1  =np.repeat(X1, 9, axis = 0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    print("Start training:", datetime.now())
    if not starting_model:
        model = LogisticModel()
    else:
        model = starting_model
    model.fit(X, Y, starting_learning_rate=starting_learning_rate, epochs=epochs, show_fig=True)
    model.score(X, Y)
    print("End training:", datetime. now())
    print("W:", model.W)
    print("b:", model.b)

    return model


def predict(model):
    X, Y = getBinaryData()
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

    model = LogisticModel()
    model.load()
    model = train(starting_learning_rate=1e-6, epochs=1000, starting_model=model)
    model.save()
    # predict(model)



if __name__ == "__main__":
    main()