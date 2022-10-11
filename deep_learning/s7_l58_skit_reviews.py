from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from s4_l22_load_data import load_data

X, Y = load_data()

X, Y = shuffle(X, Y)

Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

# create the neural network
model = MLPClassifier(hidden_layer_sizes=(20,20), activation="relu", solver="adam", alpha=10e-5, max_iter=2000)

# train
model.fit(Xtrain, Ytrain)

# get train and test accuracy
train_score = model.score(Xtrain, Ytrain)
test_score = model.score(Xtest, Ytest)

print("Train accuracy: ", train_score)
print("Test accuracy: ", test_score)