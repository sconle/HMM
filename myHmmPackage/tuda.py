import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from cluster_decoder import ClusterDecoder
import hmm

class TUDA(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            n_clusters=4,
            gamma_init=None,
            decoding_mats_init=None,
            max_iter=100,
            reg_param=10e-5,
            transition_scheme=None,
            init_scheme=None,
    ):
        self.n_clusters = n_clusters  # equivalent to K= nb of states in the Matlab implementation
        self.gamma_init = gamma_init
        self.decoding_mats_init = decoding_mats_init
        self.max_iter = max_iter
        self.reg_param = reg_param  # le lambda de la normalisation L2 pour une régression linéaire (utile dans _fit_regression)
        self.transition_scheme = transition_scheme
        self.init_scheme = init_scheme

    def fit(self, X, y):

        if not self.__check(X, y):
            return

        n_samples, n_time_points, n_regions = X.shape
        _, _, n_label_features = y.shape

        decoder = ClusterDecoder(self.n_clusters, self.gamma_init, self.decoding_mats_init,
                                 self.max_iter, self.reg_param, self.transition_scheme, self.init_scheme)
        decoder.fit(X, y)
        self.decoding_mats_ = decoder.decoding_mats_
        self.gamma_ = decoder.gamma_

        self._c_step(X, y, n_samples, n_time_points, n_regions, n_label_features)
        self._d_step(X, y, n_samples, n_time_points, n_regions, n_label_features)

        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def _c_step(self, X, y, n_samples, n_time_points, n_regions, n_label_features):
        raise NotImplementedError

    def _d_step(self, X, y, n_samples, n_time_points, n_regions, n_label_features):
        raise NotImplementedError
