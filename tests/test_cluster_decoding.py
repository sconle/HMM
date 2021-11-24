import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ..myHmmPackage.cluster_decoding import cluster_decoding

class TestClusterDecoding(unittest.TestCase):
    """
    Series of tests on the function cluster_decoding.
    Tests included are :
    - cluster_decoding returns a Gamma matrix with the right dimensions
    - cluster_decoding's output is a Gamma matrix with a value of 1 and (K-1) values of 0 on each line.
    - cluster_decoding works for different numbers of states (K)
    - All cluster_methods ('regression', 'hierarchical', 'sequential')
        return the correct Gamma matrix when executed on fake data.
    """
    # Creating a fake signal to analyse et the corresponding matrix
    X = np.array()
    Y = np.array()
    T = np.array()
    fake_gamma = np.array()

    def test_gamma_value_regression(self):
        """
        Testing on fake data that the Gamma matrix returned corresponds to the fake one.
        """
        K = 5  # Number of states chosen is arbitrary
        gamma = cluster_decoding(self.X, self.Y, self.T, K)
        assert_array_equal(gamma, self.fake_gamma, err_msg="Regression method failed to produce the right Gamma matrix")


if __name__ == '__main__':
    unittest.main()
