from abc import abstractmethod
import numpy as np
from ..base_discretizer import Discretizer


class DomainAwareDiscretizer(Discretizer):
    """
    Observation discretizer interface based on fulfilling checklist of environment specific tasks.

    Args:
        discretize_dim (int): the size of the discretized dimension
        discretized_state_dim (int): the size of the (reduced) state dimension

    Example:
        Suppose we are using batch size 128, and processing Atari frames of size (3 * 40 * 40)
        If discretize_dim = 3, discretized_state_dim = 10, then the resulting discretized
        observation batch has shape (128, 10, 3).
    """
    @abstractmethod
    def create_checklist(self, obs: np.ndarray) -> np.ndarray:
        """
        Create environment specific checklist from observation.

        Args:
            obs (ndarray): high dimensional continuous observation

        Returns:
            ndarray of size self.discretized_state_dim indicating completion level of each task

        Example:
            Suppose checkpoints consist of 
                - number of keys collected (0, 1 or 2)
                - number of platforms traversed (0, 1, or 2, 3)
            
            Then discretize_dim = 4 and discretized_state_dim = 2.
            The idea is that computing information content on this space can leverage
            strong prior for determines what consistitutes true information.
            For example, in the above representation, any agent manipulation that does not change
            the number of keys collected or platforms traversed is ignored.
        """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.create_checklist(obs)
