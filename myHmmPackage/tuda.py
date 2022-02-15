import numpy as np
import hmm

class TUDA(hmm.HMM):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError