import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ClusterDecoder(BaseEstimator, RegressorMixin):
    """
    ClusterDecoder is an Estimator that performs supervised decoding with a predefined number of decoding matrices.
    A clustering method is used to choose which decoding matrix to use for each sample of each input data.

    Parameters
    ----------
    n_clusters : int, default=4
    gamma_init : ndarray of shape (n_time_points, n_clusters) or None, default=None,
    decoding_mats_init : ndarray of shape (n_clusters, n_time_points, n_label_features) or None, default=None,
    method : str, default='regression',
    measure : str, default='error',
    max_iter : int, default=100,
    reg_param : float, default=10e-5,
    transition_scheme : ndarray of shape (n_clusters, n_clusters) or None, default=None,
    init_scheme : ndarray of shape (n_clusters,) or None, default=None

    Attributes
    ----------
    gamma_ : ndarray, shape (n_time_points, n_clusters)
    decoding_mats_ : ndarray, shape (n_clusters, n_time_points, n_label_features)
    """
    def __init__(
            self,
            n_clusters=4,
            gamma_init=None,
            decoding_mats_init=None,
            method='regression',
            measure='error',
            max_iter=100,
            reg_param=10e-5,
            transition_scheme=None,
            init_scheme=None,
    ):
        self.n_clusters = n_clusters  # equivalent to K= nb of states in the Matlab implementation
        self.gamma_init = gamma_init
        self.decoding_mats_init = decoding_mats_init
        self.method = method
        self.measure = measure
        self.max_iter = max_iter
        self.reg_param = reg_param  # le lambda de la normalisation L2 pour une régression linéaire (utile dans _fit_regression)
        self.transition_scheme = transition_scheme
        self.init_scheme = init_scheme

    def fit(self, X, y):

        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_time_points, n_regions)
            The training input samples.
        y : array-like, shape (n_samples, n_time_points, n_label_features)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y, multi_output=True)  # See documentation if we want to have more than 2d inputs

        if not self.__check(X, y):
            return

        n_samples, n_time_points, n_regions = X.shape
        _, _, n_label_features = y.shape

        # Initialize gamma_, the matrix that affects a cluster for each time point in X
        if self.gamma_init is None:
            gamma = np.zeros((n_time_points, self.n_clusters))
            div = n_time_points//self.n_clusters
            for k in range(self.n_clusters):
                gamma[k*div:(k+1)*div, k] += 1
            gamma[self.n_clusters*div:, self.n_clusters-1] += 1
            self.gamma_ = gamma
        else:
            self.gamma_ = self.gamma_init

        # Perform clustering and decoding
        if self.method == 'regression':
            self._fit_regression(X, y, n_samples, n_time_points, n_regions, n_label_features)
        elif self.method == 'hierarchical':
            self._fit_hierarchical(X, y)
        elif self.method == 'sequential':
            self._fit_sequential(X, y, n_samples, n_time_points, n_regions, n_label_features)

        return self

    def predict(self, X):
        # y_predict = np.zeros((X.shape[0],X.shape[1],self.decoding_mats_.shape[2]))
        # for t in range(10): #X.shape[1]):
        #     state = self.gamma_[t,:].tolist().index(1)
        #     y_predict[:,t,:] = X[:,t,:] @ self.decoding_mats_[state,:, :]
        #     y_predict = np.round(y_predict,0).astype(int)
        # return y_predict

        # !!!il y a littéralement AUCUN 1 dans y_predict!!!
        y_predict_states = np.zeros((self.n_clusters, X.shape[0], X.shape[1], self.decoding_mats_.shape[2]))
        for state in range(self.n_clusters):
            y_predict_states[state, :, :, :] = np.round(X @ self.decoding_mats_[state, :, :], 0).astype(int)
        y_predict = np.zeros((X.shape[0], X.shape[1], self.decoding_mats_.shape[2]))
        for t in range(X.shape[1]):
            if t != X.shape[1] - 1:
                state = self.gamma_[t,:].tolist().index(1)
            y_predict[:, t, :] = y_predict_states[state, :, t, :]
        return y_predict


    def _fit_regression(self, X, y, n_samples, n_time_points, n_regions, n_label_features):
        self.transition_scheme = np.array(self.transition_scheme).astype(int)
        self.init_scheme = np.array(self.init_scheme).astype(int)

        # Initialize decoding_mats_, the array containing n_cluster matrices, each decoding data for one cluster
        if self.decoding_mats_init is None:
            decoding_mats = np.zeros((self.n_clusters, n_regions, n_label_features))
            for cluster in range(self.n_clusters):
                n_time_points_in_cluster = sum(self.gamma_[:, cluster].astype(int))
                X_star = X[:, self.gamma_[:, cluster].astype(int) == 1, :]
                y_star = y[:, self.gamma_[:, cluster].astype(int) == 1, :]
                X_star = X_star.reshape((n_time_points_in_cluster * n_samples, n_regions))
                y_star = y_star.reshape((n_time_points_in_cluster * n_samples, n_label_features))

                decoding_mats[cluster] = np.dot(np.linalg.inv(np.dot(X_star.T, X_star)
                                                + self.reg_param * np.eye(n_regions)),
                                                np.dot(X_star.T, y_star))
            self.decoding_mats_ = decoding_mats
        else:
            self.decoding_mats_ = self.decoding_mats_init

        for _ in range(self.max_iter):
            # M step
            for cluster in range(self.n_clusters):
                n_time_points_in_cluster = sum(self.gamma_[:, cluster].astype(int))
                X_star = X[:, self.gamma_[:, cluster].astype(int) == 1, :]
                y_star = y[:, self.gamma_[:, cluster].astype(int) == 1, :]
                X_star = X_star.reshape((n_time_points_in_cluster * n_samples, n_regions))
                y_star = y_star.reshape((n_time_points_in_cluster * n_samples, n_label_features))

                self.decoding_mats_[cluster] = np.dot(np.linalg.inv(np.dot(X_star.T, X_star)
                                                      + self.reg_param * np.eye(n_regions)),
                                                      np.dot(X_star.T, y_star))

            # E step
            err = np.zeros((n_time_points, self.n_clusters))
            for cluster in range(self.n_clusters):
                norm = np.linalg.norm((y - np.dot(X, self.decoding_mats_[cluster])), axis=(0, 2))
                err[:, cluster] = norm
            gamma = np.zeros((n_time_points, self.n_clusters))
            if self.init_scheme is not None:
                err[0, self.init_scheme == 0] = np.inf
            state = np.argmin(err[0])
            gamma[0, state] = 1
            for t in range(1, n_time_points):
                if self.transition_scheme is not None:
                    err[t, self.transition_scheme[state] == 0] = np.inf
                state = np.argmin(err[t])
                gamma[t, state] = 1
            if (gamma == self.gamma_).all():
                break
            self.gamma_ = gamma

    def _fit_hierarchical(self, X, y):
        # TODO
        pass

    def _fit_sequential(self, X, y, n_samples, n_time_points, n_regions, n_label_features):
        gamma = np.zeros((n_time_points,self.n_clusters))
        states_temp_delimitation = [0] + [int(i * np.round(n_time_points / self.n_clusters)) - 1 for i in range(1, self.n_clusters)] + [n_time_points - 1]
        err = 0
        decoding_mats = np.zeros((self.n_clusters, n_regions, n_label_features))

        for states in range(self.n_clusters):  # le mat_assig_states[0] = 0 c'est un peu bizarre
            gamma[states_temp_delimitation[states]:states_temp_delimitation[states +1 ], states] = 1
            Xstar = np.reshape([X[:, i, :] for i in range(len(gamma[:, states])) if gamma[i, states]], [int(sum(gamma[:, states]) * n_samples), n_regions])
            ystar = np.reshape([y[:, i, :] for i in range(len(gamma[:, states])) if gamma[i, states]], [int(sum(gamma[:, states]) * n_samples), n_label_features])
            decoding_mats[states,:,:] = np.linalg.pinv(np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar)[1])) @ (np.transpose(Xstar) @ ystar)
            err = err + np.sqrt(sum(sum((ystar - Xstar @ decoding_mats[states,:,:]) ** 2, 2)))

        err_best = err
        gamma_best = gamma
        decoding_mats_best = decoding_mats

        for iteration in range(1, self.max_iter):
            gamma = np.zeros((n_time_points, self.n_clusters)).astype(int)
            while True:
                states_temp_delimitation = np.cumsum(1.0 + np.random.rand(1, self.n_clusters))
                states_temp_delimitation = np.concatenate((np.array([0]), np.floor(n_time_points * states_temp_delimitation / max(states_temp_delimitation)) - 1))
                if ~any(np.asarray(states_temp_delimitation) == 0) and len(np.unique(states_temp_delimitation)) == len(states_temp_delimitation):
                    break
            err = 0

            for states in range(self.n_clusters):
                gamma[int(states_temp_delimitation[states]):int(states_temp_delimitation[states + 1]), states] = 1
                Xstar = np.reshape(X[:, gamma[:, states] == 1, :], [sum(gamma[:, states]) * n_samples, n_regions])
                ystar = np.reshape(y[:, gamma[:, states] == 1, :], [sum(gamma[:, states]) * n_samples, n_label_features])
                decoding_mats[states,:,:] = np.linalg.pinv(np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar)[1])) @ ((np.transpose(Xstar) @ ystar))
                err = err + np.sqrt(sum(sum((ystar - Xstar @ decoding_mats[states,:,:]) ** 2, 2)))

            if err < err_best:
                err_best = err
                gamma_best = gamma
                decoding_mats_best = decoding_mats

        self.gamma_ = gamma_best
        self.decoding_mats_ = decoding_mats_best

    def _fit_fixed_sequential(self, X, y):
        # TODO
        pass

    def __check(self, X, y):
        # TODO check if dimensions and values of parameters are correct
        """
        dimensions X and y == (n_samples, n_time_points, n_regions) and (n_samples, n_time_points, n_label_features)
        dimensions (and values) transition_scheme == (self.n_clusters, self.n_clusters)
        dimensions gamma_init == (n_time_points, self.n_clusters)
        dimensions decoding_mats_init == (self.n_clusters, n_regions, n_label_features)
        :return: None
        """

        if len(X.shape) != 3:
            print("X need to be in 3 dimensions (n_samples, n_time_points, n_regions)")
            return False

        if len(y.shape) != 3:
            print("y need to be in 3 dimensions (n_samples, n_time_points, n_label_features)")
            return False

        if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
            print("The first two dimensions of X and y must be the same")
            return False

        if self.transition_scheme is not None and self.transition_scheme.shape != (self.n_clusters, self.n_clusters):
            print("transition_scheme's dimensions need to be: (n_clusters, n_clusters)")
            return False

        if self.gamma_init is not None and self.gamma_init.shape != (X.shape[1], self.n_clusters):
            print("gamma_init's dimensions need to be: (n_time_points, n_clusters)")
            return False

        if self.decoding_mats_init is not None and self.decoding_mats_init.shape != (self.n_clusters, X.shape[2], y.shape[2]):
            print("decoding_mats_init's dimensions need to be: (n_samples, n_time_points, n_label_features)")
            return False

        return True
