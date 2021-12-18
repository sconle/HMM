import unittest
import numpy as np
from numpy.testing import assert_array_equal
from ..myHmmPackage.cluster_decoding import cluster_decoding


def sinusoidal_function(amplitude, frequency, time):
    return amplitude * np.sin(frequency*time)


def burst_mask(minmax_burst_duration, time_steps_number, nb_tries, max_amplitude):
    """
    Creates a mask to be applied to a signal to keep only some random parts of it over time.

    :param minmax_burst_duration: list containing min and max duration of a single burst
    :param time_steps_number: number of measures taken during duration
    :param nb_tries: number of times the fake experiment was repeated
    :param max_amplitude: amplitude of the signal
    :return:
    """
    mask = np.zeros((time_steps_number, nb_tries), dtype=int)
    for try_number in range(nb_tries):
        step = 0
        while step < time_steps_number:
            burst_duration = np.random.randint(minmax_burst_duration[0], minmax_burst_duration[1] + 1)
            burst_value = np.random.randint(0, 2) * np.random.randint(1, max_amplitude + 1)
            nb_steps = 0
            while nb_steps < burst_duration and step + nb_steps < time_steps_number:
                mask[step + nb_steps, try_number] = burst_value
                nb_steps += 1
            step += burst_duration
    return mask


def fake_signal_generation(nb_states, nb_tries, duration, time_steps_number):
    """
    Creates fake data to test cluster decoding function.

    :param nb_states: number of different states expected in signal
    :param nb_tries: number of times the fake experiment was repeated
    :param duration: duration of the fake experiment
    :param time_steps_number: number of measures taken during duration
    :return: X, Y, T, Gamma
    """
    minimal_burst_duration = time_steps_number // duration + 1
    frequency = np.random.randint(1, 50, nb_states)
    amplitude = np.random.randint(1, 10, nb_states)
    T = np.linspace(0, duration, time_steps_number)
    X = np.array([sinusoidal_function(amplitude, frequency, T) for line in range(nb_tries)])
    Y = np.array([])

    fake_gamma = np.array([])
    return X, Y, T, fake_gamma


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
    # Creating a fake signal to analyse and the corresponding Gamma matrix
    X = np.array([])
    Y = np.array([])
    T = np.array([])
    fake_gamma = np.array([])

    def test_gamma_value_regression(self):
        """
        Testing on fake data that the Gamma matrix returned corresponds to the fake one.
        """
        print(burst_maskbis([5, 10], 1000, 2, 10))
        K = 5  # Number of states chosen is arbitrary
        gamma = cluster_decoding(self.X, self.Y, self.T, K)
        assert_array_equal(gamma, self.fake_gamma, err_msg="Regression method failed to produce the right Gamma matrix")


if __name__ == '__main__':
    unittest.main()
