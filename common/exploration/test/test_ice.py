import unittest
from exploration.src.intrinsic_rewards import ICE
from exploration.src.discretizers import UniformDiscretizer
from exploration.src.information import NormDeltaInformation
import numpy as np
import torch


class ICETest(unittest.TestCase):
    batch_size, image_size, discretize_dim = 2, 2, 3
    beta = 0.5
    device = "cpu"

    def setUp(self) -> None:
        np.random.seed(0)
        discretizer = UniformDiscretizer(self.discretize_dim)
        self.ice = ICE(discretizer, self.beta, self.device)
        self.ice.reset(self.gen_obs())

    def gen_obs(self):
        return np.random.rand(self.batch_size, self.image_size, self.image_size)

    def test_reset(self):
        assert self.ice.batch_size == self.batch_size
        assert isinstance(self.ice.entropy, NormDeltaInformation)
        expected_count_matrix = torch.Tensor([
            [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
        ])
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)

    def test_call_some_done(self):
        obs = self.gen_obs()
        dones = np.array([0, 1])
        reward = self.ice(obs, dones)

        expected_count_matrix = torch.Tensor([
            [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 2, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]],
        ])
        expected_reward = np.array([1.49999, 0])
        assert (self.ice.entropy.matrix.sum(-1)[0] == 2).all() # 1 count from reset, 1 count from update
        assert (self.ice.entropy.matrix.sum(-1)[1] == 1).all() # 1 count from reset
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)
        np.testing.assert_allclose(reward, expected_reward, atol=1e-5, rtol=1e-5)

    def test_call_no_done(self):
        obs = self.gen_obs()
        dones = np.array([0, 0])
        reward = self.ice(obs, dones)
        expected_count_matrix = torch.Tensor([
            [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 2, 0]],
            [[0, 2, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]],
        ])
        expected_reward = np.array([1.49999, 1.49999])
        assert (self.ice.entropy.matrix.sum(-1) == 2).all() # 1 count from reset, 1 from update
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)
        np.testing.assert_allclose(reward, expected_reward, atol=1e-5, rtol=1e-5)

    def test_call_no_done_multiple_iterations(self, num_iterations=10):
        dones = np.array([0, 0])
        rewards = []
        for _ in range(num_iterations):
            obs = self.gen_obs()
            rewards.append(self.ice(obs, dones))
        rewards = np.array(rewards)

        expected_count_matrix = torch.Tensor([
            [[5, 4, 2], [1, 8, 2], [3, 3, 5], [4, 4, 3]],
            [[2, 5, 4], [5, 3, 3], [5, 4, 2], [6, 1, 4]],
        ])
        assert (self.ice.entropy.matrix.sum(-1) == num_iterations+1).all() # 1 count from reset, 10 from update
        expected_rewards = np.array([
            [1.4999977, 1.4999977 ],
            [0.6699233, 0.33659077],
            [0.3300743, -0.02531338],
            [-0.00712132, 0.1570884 ],
            [0.15499067, 0.17950237],
            [-0.02418184, 0.25115705],
            [0.2591672, 0.28725028],
            [-0.04409242, 0.26359582],
            [0.00429177, 0.01978374],
            [0.00835562, -0.04390788],
        ])
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)
        np.testing.assert_allclose(rewards, expected_rewards, atol=1e-5, rtol=1e-5)

    def test_get_episode_trajectory(self):
        info_dict = self.ice.get_episode_information()
        assert "Trajectory Information (bits)" in info_dict
        np.testing.assert_allclose(
            info_dict["Trajectory Information (bits)"],
            np.array([0, 0]),
            atol=1e-5, rtol=1e-5,
        )
