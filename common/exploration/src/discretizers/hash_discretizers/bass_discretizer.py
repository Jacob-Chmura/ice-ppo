import numpy as np
from .base_hash_discretizer import HashDiscretizer

class BassDiscretizer(HashDiscretizer):
    """
    Basic Abstraction of ScreenShots Discretizer

    Args:
        discretized_state_dim (int): the size of the (reduced) state dimension
        original_state_dim (int): the size of the original state dimension
        cell_size: the kernel size of the average pooling operation
        num_bins: the number of bins to discretize pooling to
    """
    def __init__(self, discretized_state_dim: int, original_state_dim: int, cell_size: int, num_bins: int):
        hash_dim = original_state_dim // (cell_size * cell_size) # assume strided pools fit evenly
        super().__init__(discretized_state_dim, hash_dim)
        self.cell_size = cell_size
        self.num_bins = num_bins

    def hash(self, obs: np.ndarray) -> np.ndarray:
        obs = self._avg_pool(obs)
        obs = (self.num_bins * obs).astype(int)
        return obs

    def _avg_pool(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute average channel-wise intensity within each cell of the observation.
        
        Args:
            obs (ndarray): high-dimensional continuous observation

        Returns:
            average pooling of input observation
        """
        obs = obs.transpose(2, 3, 0, 1)
        width_in, height_in = obs.shape[:2]
        width_out, height_out = width_in // self.cell_size, height_in // self.cell_size
        out_shape = (width_out, self.cell_size, height_out, self.cell_size) + obs.shape[2:]
        obs = obs[:width_in*width_out, :height_in*height_out, ...]
        obs = np.nanmean(obs.reshape(out_shape), axis=(1, 3))
        return obs.transpose(2, 3, 0, 1)
