from collections import deque
import random
import torch
import numpy as np
from .ice import ICE
from ..information import NormDeltaInformation
from ..discretizers import Discretizer

class MICE(ICE):
    """
    Args:
        discretizer (Discretizer): the discretizer to use to preprocess continuous observations
        beta_entropy (float): intrinsic entropy reward coefficient
        beta_mutual (float): intrinsic mutual information penalty coefficient
        buffer_size (int): size of the replay buffer to sample trajectory snapshots from
        device (str): the device on which to perform computation
    """
    def __init__(
        self,
        discretizer: Discretizer,
        beta_entropy: float,
        beta_mutual: float,
        buffer_size: int=1000,
        device: str="cuda"
    ):
        super().__init__(discretizer, beta_entropy, device)
        self.beta_mutual = beta_mutual
        self.buffer = deque(maxlen=buffer_size)

    def _init_variables(self, obs: np.ndarray):
        super()._init_variables(obs)
        self.mutual = NormDeltaInformation(
            self.batch_size,
            self.discretizer.discretized_state_dim,
            self.discretizer.discretize_dim,
            self.device,
            self.beta_mutual,
        )

    def __call__(self, obs: np.ndarray, dones: np.ndarray, init: bool=False):
        dones_idx = self._update_entropy(obs, dones, init)
        self.buffer.extend(torch.split(self.entropy.matrix, 1))
        torch.vstack(random.choices(self.buffer, k=self.batch_size), out=self.mutual.matrix)
        torch.add(self.mutual.matrix, self.entropy.matrix, out=self.mutual.matrix)
        reward = self.entropy() - self.mutual()
        reward[dones_idx] = 0
        return reward
