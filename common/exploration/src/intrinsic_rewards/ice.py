import numpy as np
import torch
from ..information import NormDeltaInformation
from ..discretizers import Discretizer


class ICE:
    """
    Information Content based Exploration

    Args:
        discretizer (Discretizer): the discretizer to use to preprocess continuous observations
        beta (float): intrinsic entropy reward coefficient
        device (str): the device on which to perform computation
    """
    def __init__(self, discretizer: Discretizer, beta: float, device: str="cuda"):
        self.discretizer = discretizer
        self.beta = beta
        self.device = torch.device(device)

    def reset(self, obs: np.ndarray) -> None:
        """
        Reset internal arrays and compute entropy of first observation after environment reset.

        Args:
            obs (np.ndarray): batch of the first observations from the environments
        """
        self._init_variables(obs)
        self(obs, dones=np.ones((self.batch_size)), init=True)

    def __call__(self, obs: np.ndarray, dones: np.ndarray, init: bool=False) -> np.ndarray:
        """
        Compute intrinsic ICE reward.

        Args:
            obs (np.ndarray): batch of current observations from the environments
            dones (np.ndarray): batch of done flags from the environments
            init (bool): flag that determines whether we are calling this method from reset.
        """
        dones_idx = self._update_entropy(obs, dones, init)
        reward = self.entropy()
        reward[dones_idx] = 0
        return reward

    def get_episode_information(self):
        """
        Return a dictionary of episode statistics that we want to monitor.
        """
        return {
            "Trajectory Information (bits)": self.entropy.last_entropy,
        }

    def _init_variables(self, obs: np.ndarray):
        """
        Initialize attributes that depend on observation shape.

        Args:
            obs (np.ndarray): batch of current observations from the environments
        """
        self.batch_size = obs.shape[0]
        self.entropy = NormDeltaInformation(
            self.batch_size,
            self.discretizer.discretized_state_dim,
            self.discretizer.discretize_dim,
            self.device,
            self.beta,
        )

    def _update_entropy(self, obs: np.ndarray, dones: np.ndarray, init: bool):
        """
        Update internal arrays given new observations from the environment.

        Args:
            obs (np.ndarray): batch of current observations from the environments
            dones (np.ndarray): batch of done flags from the environments
            init (bool): flag that determines whether we are calling this method from reset.
        """
        obs = self.discretizer(obs)
        dones_idx = np.where(dones == 1)
        if dones.any() and not init:
            self.entropy.matrix[dones_idx] = 0
        self.entropy.update(obs)
        return dones_idx
