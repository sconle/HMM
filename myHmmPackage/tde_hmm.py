import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from hmmlearn.hmm import GaussianHMM
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TDE_HMM(GaussianHMM):
    """
    TDE_HMM is an Estimator that performs supervised decoding with a predefined number of decoding matrices.
    A clustering method is used to choose which decoding matrix to use for each sample of each input data.

    Attributes
    ----------
     n_features : int
        Dimensionality of the Gaussian emissions.
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    covars_ : array
        Covariance parameters for each state.
        The shape depends on :attr:`covariance_type`:
        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    """

    def __init__(
            self,
            n_components=3, covariance_type='full',
            min_covar=1e-3,
            startprob_prior=1.0, transmat_prior=1.0,
            means_prior=0, means_weight=0,
            covars_prior=1e-2, covars_weight=1,
            algorithm="viterbi", random_state=None,
            n_iter=10, tol=1e-2, verbose=False,
            params="stmc", init_params="stmc",
    ):
        """
            Parameters
    ----------
    n_components : int
            Number of states.
        covariance_type : {"sperical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:
            * "spherical" --- each state uses a single variance value that
              applies to all features (default).
            * "diag" --- each state uses a diagonal covariance matrix.
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all states use **the same** full covariance matrix.
        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting. Defaults to 1e-3.
        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.
        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.
        means_prior, means_weight : array, shape (n_components, ), optional
            Mean and precision of the Normal prior distribtion for
            :attr:`means_`.
        covars_prior, covars_weight : array, shape (n_components, ), optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`.
            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.
        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.
        random_state: RandomState or an int seed, optional
            A random number generator instance.
        n_iter : int, optional
            Maximum number of iterations to perform.
        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.
        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.
        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, and 'c' for
            covars.  Defaults to all parameters.
        implementation: string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """

        super().__init__(n_components=n_components,
                         covariance_type=covariance_type, min_covar=min_covar,
                         startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                         means_prior=means_prior, means_weight=means_weight,
                         covars_prior=covars_prior, covars_weight=covars_weight,
                         algorithm=algorithm, random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params)

    def fit(self, X, y=None):
        X = self.__signal_crante(X)
        super().fit(X)

        return self

    def predict_proba(self, X, y=None):
        n_samples, n_time_points, _ = X.shape
        n_fenetre = 20
        X = self.__signal_crante(X, n_fenetre)
        posteriors = super().predict_proba(X)
        padding = np.zeros((n_fenetre, self.n_components))
        posteriors = np.concatenate((posteriors, padding))
        posteriors = np.reshape(posteriors, (n_samples, n_time_points, posteriors.shape[1]))
        return posteriors

    def __signal_crante(self, X, n_fenetre=20):
        # On récupère un signal
        X = X[:, :, 0]

        # On concatène pour obtenir qu'un seul array
        X = np.reshape(X, X.shape[0] * X.shape[1])
        X = np.reshape(X, -1)

        # On découpe le signal en plusieurs sous signaux décalés d'un cran
        n = len(X)
        signal_crante = np.ones((n - n_fenetre, n_fenetre))

        for i in range(n - n_fenetre):
            signal_crante[i] = X[i:i + n_fenetre]

        return signal_crante
