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
    max_iter : int, default=100

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
    ):
        self.n_clusters = n_clusters  # equivalent to K= nb of states in the Matlab implementation
        self.gamma_init = gamma_init
        self.decoding_mats_init = decoding_mats_init
        self.method = method
        self.measure = measure
        self.max_iter = max_iter
        self.reg_param = reg_param  # le lambda de la normalisation L2 pour une régression linéaire (utile dans _fit_regression)
        # TODO : autres params tels que Pstructure et Pistructure et surtout T

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

        n_samples, n_time_points, n_regions = X.shape
        _, _, n_label_features = y.shape

        # Initialize gamma_, the matrix that affects a cluster for each time point in X
        if self.gamma_init is None:
            gamma = np.zeros((n_time_points, self.n_clusters))
            div = n_time_points//self.n_clusters
            for k in range(self.n_clusters):
                gamma[k*div:(k+1)*div, k] += 1
            gamma[self.n_clusters:, self.n_clusters-1] += 1
            self.gamma_ = gamma
        else:
            # TODO : check if gamma_init has the right dimensions
            self.gamma_ = self.gamma_init

        # Initialize decoding_mats_, the array containing n_cluster matrices, each decoding data for one cluster
        if self.decoding_mats_init is None:
            decoding_mats = np.zeros((self.n_clusters, n_regions, n_label_features))
            # TODO : regressions for each cluster
            self.decoding_mats_ = decoding_mats
        else:
            # TODO : check if decoding_mats_init has the right dimensions
            self.decoding_mats_ = self.decoding_mats_init

        # Perform clustering and decoding
        if self.method == 'regression':
            self._fit_regression(X, y)
        elif self.method == 'hierarchical':
            self._fit_hierarchical(X, y)
        elif self.method == 'sequential':
            self._fit_sequential(X, y)

        return self

    def predict(self, X):
        # TODO : perform clustering wrt maximum likelihood, then return y, result of the decoding analysis.
        pass

    def _fit_regression(self, X, y, n_regions, n_label_features):
        # Initialize decoding_mats_, the array containing n_cluster matrices, each decoding data for one cluster
        if self.decoding_mats_init is None:
            decoding_mats = np.zeros((self.n_clusters, n_regions, n_label_features))
            # TODO : regressions for each cluster
            self.decoding_mats_ = decoding_mats
        else:
            # TODO : check if decoding_mats_init has the right dimensions
            self.decoding_mats_ = self.decoding_mats_init

        # M step
        for cluster in self.n_clusters:
            n_time_points_in_cluster = sum(self.gamma_[:, cluster])
            Xstar = X[:, self.gamma_[:, cluster], :]
            Xstar = Xstar.reshape((n_time_points_in_cluster*self.n_clusters, n_regions))
            ystar = y[:, self.gamma_[:, cluster], :]
            ystar = ystar.reshape((n_time_points_in_cluster * self.n_clusters, n_label_features))

            self.decoding_mats_[cluster] = np.dot((np.dot(Xstar.T, Xstar) + self.reg_param*np.eye(n_regions)), np.inv(np.dot(Xstar.T, ystar)))

        # TODO : Faire la boucle EM
        pass

    def _fit_hierarchical(self, X, y):
        # TODO
        pass

    def _fit_sequential(self, X, y, n_samples, n_time_points, n_regions, n_label_features):
        mat_assig_states = np.zeros((n_time_points, 1))
        mat_changes_states = [0] + [int(i * np.round(n_time_points / self.n_clusters)) - 1 for i in range(1, self.n_clusters)] + [n_time_points - 1]
        err = 0

        for states in range(1, self.n_clusters + 1):  # le mat_assig_states[0] = 0 c'est un peu bizarre
            mat_assig_states[mat_changes_states[states - 1]:mat_changes_states[states]] = states
            mat_is_in_states = mat_assig_states == states
            Xstar = np.reshape([X[i, :, :] for i in range(len(mat_is_in_states)) if mat_is_in_states[i]], [int(sum(mat_is_in_states) * n_samples), n_regions])
            ystar = np.reshape([y[i, :, :] for i in range(len(mat_is_in_states)) if mat_is_in_states[i]], [int(sum(mat_is_in_states) * n_samples), n_label_features])

            decoding_mats = np.linalg.pinv(np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar)[1])) @ (np.transpose(Xstar) @ ystar)
            err = err + np.sqrt(sum(sum((ystar - Xstar @ decoding_mats) ** 2, 2)))

        err_best = err
        mat_assig_states_best = mat_assig_states
        decoding_mats_best = decoding_mats

        for iteration in range(1, self.max_iter):
            mat_assig_states = np.zeros((n_time_points, 1))
            while True:
                mat_changes_states = np.cumsum(1.0 + np.random.rand(1, self.n_clusters))
                mat_changes_states = np.concatenate((np.array([0]), np.floor(n_time_points * mat_changes_states / max(mat_changes_states)) - 1))
                if ~any(np.asarray(mat_changes_states) == 0) and len(np.unique(mat_changes_states)) == len(mat_changes_states):
                    break
            err = 0

            for states in range(1, self.n_clusters + 1):
                mat_assig_states[int(mat_changes_states[states-1]):int(mat_changes_states[states])] = states
                mat_is_in_states = mat_assig_states == states
                mat_is_in_states = np.transpose(np.array(mat_is_in_states))[0]
                                                        #je ne comprends pas totalement pourquoi mais visiblement il faut transposer mat_is_in_states et une
                                                        #fois transposé, il faut prendre le premier élément de la liste car ça devient une liste de liste
                Xstar = np.reshape(X[mat_is_in_states, :, :], [sum(mat_is_in_states) * n_samples, n_regions])
                ystar = np.reshape(y[mat_is_in_states, :, :], [sum(mat_is_in_states) * n_samples, n_label_features])
                decoding_mats = np.linalg.pinv(np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar)[1])) @ ((np.transpose(Xstar) @ ystar))
                err = err + np.sqrt(sum(sum((ystar - Xstar @ decoding_mats) ** 2, 2)))

            if err < err_best:
                err_best = err
                mat_assig_states_best = mat_assig_states
                decoding_mats_best = decoding_mats

        mat_assig_states = mat_assig_states_best
        self.decoding_mats_ = decoding_mats_best
        # TODO
        pass

    def _fit_fixed_sequential(self, X, y):
        # TODO
        pass
