from exploration.src.intrinsic_rewards import MICE
from exploration.test.test_ice import ICETest
from exploration.src.discretizers import UniformDiscretizer
from exploration.src.information import NormDeltaInformation
import numpy as np
import torch
import random


class MICETest(ICETest):
    beta_mutual = 1
    buffer_size = 10

    def setUp(self) -> None:
        random.seed(0)
        np.random.seed(0)
        discretizer = UniformDiscretizer(self.discretize_dim)
        self.ice = MICE(discretizer, self.beta, self.beta_mutual, self.buffer_size, self.device)
        self.ice.reset(self.gen_obs())

    def test_reset(self):
        super().test_reset()
        assert isinstance(self.ice.mutual, NormDeltaInformation)
        assert len(self.ice.buffer) == self.ice.batch_size

    def test_call_some_done(self):
        obs = self.gen_obs()
        dones = np.array([0, 1])
        reward = self.ice(obs, dones)
        expected_count_matrix = torch.Tensor([
            [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 2, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]],
        ])
        expected_reward = np.array([-0.839847, 0])
        assert len(self.ice.buffer) == 2 * self.ice.batch_size
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
        expected_reward = np.array([-1.311274, -1.49999])
        assert len(self.ice.buffer) == 2 * self.ice.batch_size
        assert (self.ice.entropy.matrix.sum(-1) == 2).all() # 1 count from reset, 1 from update
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)
        np.testing.assert_allclose(reward, expected_reward, atol=1e-5, rtol=1e-5)

    def test_call_no_done_multiple_iterations(self, num_iterations=10):
        dones = np.array([0, 0])
        rewards = []
        for i in range(num_iterations):
            if i > 4:
                expected_buffer_size = 10
            else:
                expected_buffer_size = (i + 1) * self.ice.batch_size
            assert len(self.ice.buffer) == expected_buffer_size
            obs = self.gen_obs()
            rewards.append(self.ice(obs, dones))
        rewards = np.array(rewards)

        expected_count_matrix = torch.Tensor([
            [[5, 4, 2], [1, 8, 2], [3, 3, 5], [4, 4, 3]],
            [[2, 5, 4], [5, 3, 3], [5, 4, 2], [6, 1, 4]],
        ])

        assert (self.ice.entropy.matrix.sum(-1) == num_iterations+1).all() # 1 count from reset, 10 from update
        expected_rewards = np.array([
            [-1.3112744,  -1.4999977 ],
            [ 0.18546295, -1.9591432 ],
            [ 0.625813,   -0.08964682],
            [ 0.00712132,  1.5804242 ],
            [-0.27813625, -0.17950237],
            [ 0.14732742, -0.25115705],
            [-0.2591672,  -1.0144656 ],
            [-0.47705388,  0.46361947],
            [ 0.5168545,  -0.01978374],
            [-0.47748184,  0.04390788],
        ])
        torch.testing.assert_close(self.ice.entropy.matrix, expected_count_matrix)
        np.testing.assert_allclose(rewards, expected_rewards, atol=1e-5, rtol=1e-5)

    def test_get_episode_trajectory(self):
        super().test_get_episode_trajectory()
