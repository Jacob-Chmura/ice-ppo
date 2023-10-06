import unittest
from exploration.src.discretizers import BassDiscretizer
import numpy as np

class BassDiscretizerTest(unittest.TestCase):
    discretized_state_dim = 3
    original_state_dim = 3 * 4 * 4 # C, W, H
    cell_size = 2
    num_bins = 10

    def setUp(self):
        self.discretizer = BassDiscretizer(self.discretized_state_dim, self.original_state_dim, self.cell_size, self.num_bins)

    def _gen_obs(self):
        batch_size, width, height, channels = 2, 4, 4, 3
        obs = np.arange(batch_size * width * height * channels)
        obs = (obs / obs.max()).reshape(batch_size, width, height, channels)
        obs = np.around(obs, 1)
        return obs

    def test_init(self):
        assert self.discretizer.hash_dim == self.original_state_dim // (self.cell_size * self.cell_size)
        assert self.discretizer.cell_size == self.cell_size
        assert self.discretizer.num_bins == self.num_bins
        assert self.discretizer.discretized_state_dim == self.discretized_state_dim
        assert self.discretizer.discretize_dim == 2
        assert self.discretizer.projector.shape == (self.discretizer.hash_dim, self.discretizer.discretized_state_dim)

    def test_call(self):
        obs = self._gen_obs()
        discretized_obs = self.discretizer(obs)
        assert set(list(discretized_obs.ravel())).issubset({0, 1})

    def test_avg_pool(self):
        obs = self._gen_obs()
        pooled_obs = self.discretizer._avg_pool(obs)
        expected_pool_obs = [0.075, 0.075, 0.1, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.375, 0.425, 0.425, 0.575, 0.575, 0.625, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85, 0.9, 0.925, 0.925]
        assert pooled_obs.shape == (2, 2, 2, 3)
        np.testing.assert_allclose(pooled_obs.ravel(), expected_pool_obs)

    def test_hash(self):
        obs = self._gen_obs()
        hashed_obs = self.discretizer.hash(obs)
        expected_hashed_obs = [0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 8, 8, 8, 9, 9, 9]
        assert hashed_obs.shape == (2, 2, 2, 3)
        np.testing.assert_allclose(hashed_obs.ravel(), expected_hashed_obs)
