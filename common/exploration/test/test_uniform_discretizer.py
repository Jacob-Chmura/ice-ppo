import unittest
from exploration.src.discretizers import UniformDiscretizer
import numpy as np

class UniformDiscretizerTest(unittest.TestCase):
    discretize_dim = 3
    device = "cpu"

    def setUp(self):
        self.discretizer = UniformDiscretizer(self.discretize_dim)

    def test_call(self):
        obs = np.array([
            [0.1, 0.2, 0.5, 0],
            [0.9, 0.99, 0.4, 0.5],
        ])
        expected_discretized_obs = np.array([
            [0, 0, 1, 0],
            [2, 2, 1, 1],
        ])
        discretized_obs = self.discretizer(obs)
        np.testing.assert_allclose(discretized_obs, expected_discretized_obs)
