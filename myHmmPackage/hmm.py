import numpy as np
from sklearn.base import BaseEstimator

class HMM(BaseEstimator):
    def __init__(self):

    def fit(self, X, y):        
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

hmm = HMM(K, pas)

hmm.fit(X)

hmm.predict(Xtest)