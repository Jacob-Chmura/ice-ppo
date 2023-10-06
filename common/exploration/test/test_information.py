import unittest
from exploration.src.information import Information
from exploration.src.information import NormDeltaInformation
from exploration.test.helpers import assert_equal
import torch
import numpy as np


class InformationTest(unittest.TestCase):
    batch_size = 2
    state_dim = 4
    discretize_dim = 3
    device = "cpu"

    def setUp(self):
        self.info = Information(self.batch_size, self.state_dim, self.discretize_dim, self.device)

    def test_init(self):
        zeros = torch.zeros(self.batch_size, self.state_dim, self.discretize_dim)
        assert_equal(self.info.matrix, zeros)
        assert_equal(self.info.p_matrix, zeros)
        assert_equal(self.info.logp_matrix, zeros)

        batch_idx = torch.LongTensor(np.arange(self.batch_size).repeat(self.state_dim), device=self.device)
        state_idx = torch.LongTensor(np.tile(np.arange(self.state_dim), self.batch_size), device=self.device)
        assert (batch_idx == self.info._batch_idx).all()
        assert (state_idx == self.info._state_idx).all()

    def test_update(self):
        obs = np.array([
            [[0], [2], [1], [0]],
            [[1], [1], [2], [0]],
        ])
        expected_matrix_after_update = torch.Tensor([
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        ])
        self.info.update(obs)
        assert_equal(self.info.matrix, expected_matrix_after_update)

    def test_call(self):
        obs = np.array([
            [[0], [2], [1], [0]],
            [[1], [1], [2], [0]],
        ])
        self.info.update(obs)
        entropy = self.info()
        np.testing.assert_allclose(entropy, [0, 0], rtol=1e-5, atol=1e-5)

        obs = np.array([
            [[1], [2], [0], [0]],
            [[0], [2], [1], [1]],
        ])
        self.info.update(obs)
        entropy = self.info()
        np.testing.assert_allclose(entropy, [2, 4], rtol=1e-5, atol=1e-5)



class NormDeltaInformationTest(InformationTest):
    norm_factor = 0.5
    def setUp(self):
        self.info = NormDeltaInformation(self.batch_size, self.state_dim, self.discretize_dim, self.device, self.norm_factor)

    def test_init(self):
        super().test_init()
        assert self.info.norm_factor == self.norm_factor
        assert self.info.last_entropy is None

    def test_update(self):
        super().test_update()

    def test_call(self):
        obs = np.array([
            [[0], [2], [1], [0]],
            [[1], [1], [2], [0]],
        ])
        self.info.update(obs)
        entropy = self.info()
        np.testing.assert_allclose(entropy, [0, 0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(self.info.last_entropy, [0, 0], rtol=1e-5, atol=1e-5)

        obs = np.array([
            [[1], [2], [0], [0]],
            [[0], [2], [1], [1]],
        ])
        self.info.update(obs)
        entropy = self.info()
        np.testing.assert_allclose(entropy, [1, 2], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(self.info.last_entropy, [2, 4], rtol=1e-5, atol=1e-5)

        obs = np.array([
            [[2], [2], [0], [0]],
            [[1], [1], [1], [1]],
        ])
        self.info.update(obs)
        entropy = self.info()
        np.testing.assert_allclose(entropy, [0.251628, -0.163408], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(self.info.last_entropy, [2.503248, 3.673172], rtol=1e-5, atol=1e-5)
