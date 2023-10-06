from abc import abstractmethod
import numpy as np
from ..base_discretizer import Discretizer

class HashDiscretizer(Discretizer):
    """
    Observation discretizer interface based on locality sensitive hashing.

    Args:
        discretized_state_dim (int): the size of the (reduced) state dimension
        hash_dim (int): the dimension of the hash function output
    """
    def __init__(self, discretized_state_dim: int, hash_dim: int):
        discretize_dim = 2 # will binarize hashed output
        super().__init__(discretize_dim, discretized_state_dim)
        self.hash_dim = hash_dim
        self.projector = np.random.normal(size=(self.hash_dim, self.discretized_state_dim))

    @abstractmethod
    def hash(self, obs: np.ndarray) -> np.ndarray:
        """
        Hash the observation values.
        
        Args:
            obs (ndarray): high-dimensional continuous observation shape (batch_size, state_dim)
        
        Returns:
            hashed observation (ndarray) with values in self.hash_dim
        """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Discretize the observation.

        Args:
            obs (ndarray): high-dimensional continuous observation shape (batch_size, state_dim)

        Returns:
            A binarized version of obs after passing through hash function.
            Modifies the shape to size (batch_size, self.discretized_state_dim)
        """
        obs = self.hash(obs)
        obs = obs.reshape(obs.shape[0], -1) # collapse state dimension
        obs = obs @ self.projector
        obs = (obs >= 0).astype(int)
        return obs
