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


    def softmax(self, a):
        expa = np.exp(a)
        return expa/expa.sum(axis=0, keepddims=True)

    def predict(self, X, W, b):
        a = X.dot(W)+b
        return self.softmax(a)
