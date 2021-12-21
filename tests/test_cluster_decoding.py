import unittest
import numpy as np
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal
from ..myHmmPackage.cluster_decoding import cluster_decoding


def sinusoidal_function(amplitude, frequency, time):
    return amplitude * np.sin(frequency*time)


def burst_mask(minmax_burst_duration, time_steps_number, nb_tries):
    """
    Creates a mask to be applied to a signal to keep only some random parts of it over time.

    :param minmax_burst_duration: list containing min and max duration of a single burst
    :param time_steps_number: number of measures taken during duration
    :param nb_tries: number of times the fake experiment was repeated
    :return: 2 dim matrix to apply to a signal to creates bursts with variation of amplitudes
    """
    mask = np.zeros((time_steps_number, nb_tries))
    for try_number in range(nb_tries):
        step = 0
        while step < time_steps_number:
            burst_duration = np.random.randint(minmax_burst_duration[0], minmax_burst_duration[1] + 1)
            burst_value = np.random.randint(0, 2) * np.random.randint(1, 11)/10
            nb_steps = 0
            while nb_steps < burst_duration and step + nb_steps < time_steps_number:
                mask[step + nb_steps, try_number] = burst_value
                nb_steps += 1
            step += burst_duration
    return mask


def fake_signal_generation(nb_states, nb_tries, time_steps_number, time_stamps):
    """
    Creates fake data to test cluster decoding function.

    :param time_stamps: list containing the time values
    :param nb_states: number of different states expected in signal
    :param nb_tries: number of times the fake experiment was repeated
    :param duration: duration of the fake experiment
    :param time_steps_number: number of measures taken during duration
    :return: X, Y, T, Gamma
    """

    state_duration = time_steps_number // nb_states
    frequency = np.random.randint(1, 1000, nb_states)
    amplitude = np.random.randint(1, 10, nb_states)
    X = np.zeros((time_steps_number - state_duration * nb_states, nb_tries), dtype=int)
    for state_range in range(nb_states):
        t = time_stamps[(nb_states - state_range - 1)*state_duration:(nb_states - state_range)*state_duration]
        X_slice = np.array([sinusoidal_function(amplitude[state_range], frequency[state_range], t) for line in range(nb_tries)]).T
        X = np.concatenate((X_slice, X), axis=0)
    return X


def fake_gamma_generation(time_steps_number, nb_states):
    state_duration = time_steps_number // nb_states
    gamma = np.zeros((time_steps_number, nb_states), dtype=int)
    gamma_ending = np.ones((time_steps_number - state_duration * nb_states, nb_states), dtype=int)
    for state in range(nb_states):
        gamma[state * state_duration: (state + 1) * state_duration, state] = 1
    gamma = np.concatenate((gamma, gamma_ending), axis=0)
    return gamma


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

    # Parameters of the fake experiment
    nb_of_measures = 1000
    nb_of_tries = 2
    min_max_burst_duration = [10, 50]
    nb_states = 4
    duration = 10

    # Creating a fake signal to analyse and the corresponding Gamma matrix
    T = np.linspace(0, duration, nb_of_measures)
    fake_signal = fake_signal_generation(nb_states, nb_of_tries, nb_of_measures, T)
    mask = burst_mask(min_max_burst_duration, nb_of_measures, nb_of_tries)
    Y = (mask > 0).astype(int)
    noise = 0.9*np.random.normal(0, 1, (nb_of_measures, nb_of_tries))
    X = fake_signal * mask + noise
    no_noise_X = fake_signal * mask
    fake_gamma = fake_gamma_generation(nb_of_measures, nb_states)

    def test_gamma_value_regression(self):
        """
        Testing on fake data that the Gamma matrix returned corresponds to the fake one.
        """

        plt.plot(self.T, self.X.T[0], label='X')
        plt.plot(self.T, self.Y.T[0], label='Y')
        plt.legend(['X', 'Y'])
        plt.show()
        plt.plot(self.T, self.no_noise_X.T[0], label='X')
        plt.plot(self.T, self.Y.T[0], label='Y')
        plt.legend(['X', 'Y'])
        plt.show()
        K = self.nb_states  # Number of states chosen is arbitrary
        gamma = cluster_decoding(self.no_noise_X, self.Y, self.T, K)
        assert_array_equal(gamma, self.fake_gamma, err_msg="Regression method failed to produce the right Gamma matrix")


if __name__ == '__main__':
    unittest.main()
