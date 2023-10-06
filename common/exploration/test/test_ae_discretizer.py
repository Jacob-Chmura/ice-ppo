
import unittest
from collections import deque
from unittest.mock import patch
from exploration.src.discretizers import AEDiscretizer
from exploration.src.discretizers.hash_discretizers.learned_discretizers.models import ConvAutoEncoder
import torch
import numpy as np


class AEDiscretizerTest(unittest.TestCase):
    discretized_state_dim = 3
    hash_dim = 512
    update_every = 5
    device = torch.device("cuda")
    noise_range = 0.3
    lmbda = 0.1

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.discretizer = AEDiscretizer(self.discretized_state_dim, self.hash_dim, self.update_every, self.device, self.noise_range, self.lmbda)

    def _gen_obs(self):
        batch_size, width, height, channels = 2, 64, 64, 3
        obs = np.arange(batch_size * width * height * channels)
        obs = (obs / obs.max()).reshape(batch_size, width, height, channels)
        obs = np.around(obs, 1)
        return obs

    def test_init(self):
        assert self.discretizer.update_every == self.update_every
        assert self.discretizer.device == self.device
        assert self.discretizer.count == 0
        assert self.discretizer.batch_size == 32
        assert self.discretizer.lmbda == self.lmbda
        assert isinstance(self.discretizer.model, ConvAutoEncoder)
        assert isinstance(self.discretizer.buffer, deque)
        assert isinstance(self.discretizer.optimizer, torch.optim.Adam)
        assert isinstance(self.discretizer.noise_distribution, torch.distributions.Uniform)

    def test_call(self):
        obs = self._gen_obs()
        discretized_obs = self.discretizer(obs)
        assert discretized_obs.shape == (obs.shape[0], self.discretized_state_dim)
        assert set(list(discretized_obs.ravel())).issubset({0, 1})
        assert self.discretizer.count == 1
        assert len(self.discretizer.buffer) == obs.shape[0]

    @patch("exploration.src.discretizers.LearnedHashDiscretizer._update")
    def test_call_with_update(self, mock_update):
        for i in range(1, self.update_every + 1):
            obs = self._gen_obs()
            discretized_obs = self.discretizer(obs)

            assert discretized_obs.shape == (obs.shape[0], self.discretized_state_dim)
            assert set(list(discretized_obs.ravel())).issubset({0, 1})
            assert self.discretizer.count == i
            assert len(self.discretizer.buffer) == i * obs.shape[0]
            if i == self.update_every:
                mock_update.assert_called()
            else:
                mock_update.assert_not_called()

    def test_hash(self):
        obs = self._gen_obs()
        hashed_obs = self.discretizer.hash(obs)
        assert hashed_obs.shape == (obs.shape[0], self.hash_dim)

    def test_loss(self):
        # set model activations
        self.discretizer.model.latent_activations = torch.rand(2, self.hash_dim)
        inputs = torch.from_numpy(self._gen_obs())
        outputs = inputs + torch.rand(inputs.shape)
        loss = self.discretizer.loss(inputs, outputs).item()
        expected_loss = 0.3431137
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)
