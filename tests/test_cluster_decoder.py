import unittest
import numpy as np
from numpy.testing import assert_array_equal
from ..myHmmPackage.cluster_decoder import ClusterDecoder


class TestClusterDecoder(unittest.TestCase):
    """
    Series of tests on the function cluster_decoder.
    Tests included are :

    """

    def test_ClusterDecoder_init(self):
        model = ClusterDecoder()
        params_to_compare = [
            model.n_clusters,
            model.gamma_init,
            model.decoding_mats_init,
            model.method,
            model.measure,
            model.max_iter,
            model.reg_param,
            model.transition_scheme,
            model.init_scheme
        ]
        values_expected = [4, None, None, 'regression', 'error', int(1e2), 1e-5, None, None]
        assert_array_equal(params_to_compare, values_expected, err_msg="Init failed")

    def test_fit_check(self):
        """
        Checking the right error message is returned if X and y are not the right dimension.
        :return:
        """
        model = ClusterDecoder()
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        # Creating a one line differential
        y = np.ones((n_samples - 1, n_time_points, n_label_features))
        assert not model.fit(X, y)

    def test_fit_regression_gamma_dims(self):
        """
        Testing that gamma matrix returned by _fit_regression has the right dimensions.
        :return:
        """
        model = ClusterDecoder()
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        dims = gamma.shape
        assert dims == (500, model.n_clusters)

    def test_fit_regression_gamma_ones(self):
        """
        Testing that gamma matrix returned by _fit_regression only contains one instance of '1' per line.
        :return:
        """
        model = ClusterDecoder()
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        sum_rows = np.sum(gamma, axis=1)
        assert_array_equal(sum_rows, np.ones(len(sum_rows)), err_msg="Gamma matrix contains too many instances of 1s "
                                                                    "on each line")

    def test_fit_regression_states(self):
        """
        Checking that each state succeeds to the previous one in the right order.
        :return:
        """
        model = ClusterDecoder()
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        # Initializing a variable to store information on the previous instance of '1' (number of previous state)
        state = 0
        # Initializing a variable to check that the serie of state is correct
        state_serie = True
        for index in range(gamma.shape[0]):
            for column in range(gamma.shape[1]):
                if gamma[index, column] == 1:
                    if not (column == state % 3 or column == state % 3 + 1):
                        state_serie = False
                        break
                    state = column
        assert state_serie

    def test_fit_sequential_dims(self):
        """
        Testing that gamma matrix returned by _fit_sequential has the right dimensions.
        :return:
        """
        model = ClusterDecoder()
        model.method = 'sequential'
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        dims = gamma.shape
        assert dims == (500, model.n_clusters)

    def test_fit_sequential_ones(self):
        """
        Testing that gamma matrix returned by _fit_sequential only contains one instance of '1' per line.
        :return:
        """
        model = ClusterDecoder()
        model.method = 'sequential'
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        sum_rows = np.sum(gamma, axis=1)
        assert_array_equal(sum_rows, np.ones(len(sum_rows)), err_msg="Gamma matrix contains too many instances of 1s "
                                                                     "on each line")

    def test_fit_sequential_state_order(self):
        """
        Checking that each state succeeds to the previous onein the right order.
        :return:
        """
        model = ClusterDecoder()
        model.method = 'sequential'
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        gamma = model.fit(X, y).gamma_
        # Initializing a variable to store information on the previous instance of '1' (number of previous state)
        state = 0
        # Initializing a variable to check that the serie of state is correct
        state_serie = True
        for index in range(gamma.shape[0]):
            for column in range(gamma.shape[1]):
                if gamma[index, column] == 1:
                    if not (column == state or column == state + 1):
                        state_serie = False
                        break
                    state = column
        assert state_serie

    def test_predict(self):
        """
        Checking dimensions of the y_predict matrix returned by predict method.
        :return:
        """
        model = ClusterDecoder()
        n_samples = 1000
        n_time_points = 500
        n_regions = 8
        n_label_features = 5
        X = np.ones((n_samples, n_time_points, n_regions))
        y = np.ones((n_samples, n_time_points, n_label_features))
        model.fit(X, y)
        y_predict = model.predict(X)
        assert y_predict.shape == (X.shape[0], X.shape[1], model.decoding_mats_.shape[2])


if __name__ == '__main__':
    unittest.main()
