from abc import ABCMeta
from abc import abstractmethod
import numpy as np


class Discretizer(metaclass=ABCMeta):
    """
    Observation discretizer interface.

    Args:
        discretize_dim: the number of discrete states in the categorical discretized distribution
        discretized_state_dim: the size of the discretized state dimension
    """
    def __init__(self, discretize_dim: int, discretized_state_dim: int):
        self.discretize_dim = discretize_dim
        self.discretized_state_dim = discretized_state_dim

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Discretize the observation.

        Args:
            obs (ndarray): high-dimensional continuous observation shape (batch_size, state_dim)

        Returns:
            A discretized version of obs with at most self.discretize_dim unique values.
            Retains the same shape of (batch_size, state_dim)
        """
