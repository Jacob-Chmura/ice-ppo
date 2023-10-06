import numpy as np
import torch
import torch.nn.functional as f

class Information:
    """
    Memory aware batched computation of shannon entropy.

    Args:
        batch_size: size of axis of independent trajectory samples
        state_dim: size of axis of state dimension
        discretize_dim: size of axis of discretized dimension
        device: torch device on which to perform computation

    Example:
        If (batch_size, state_dim, discretize_dim) = (128, 100, 16), then
        we compute entropy for the 128 samples, where the entropy is based on a 16-state
        discrete distribution, and we assume independence on each of the 100 state dimensions,
        resulting on a float tensor of size (128,).
    """
    def __init__(self, batch_size: int, state_dim: int, discretize_dim: int, device: torch.device):
        self.device = device
        matrix_shapes = (batch_size, state_dim, discretize_dim)
        self.matrix = torch.zeros(matrix_shapes, dtype=torch.float32, device=device)
        self.p_matrix = torch.zeros(matrix_shapes, dtype=torch.float32, device=device)
        self.logp_matrix = torch.zeros(matrix_shapes, dtype=torch.float32, device=device)
        self._batch_idx = torch.tensor(np.arange(batch_size).repeat(state_dim), dtype=torch.int64, device=device)
        self._state_idx = torch.tensor(np.tile(np.arange(state_dim), batch_size), dtype=torch.int64, device=device)

    def update(self, obs: np.ndarray) -> None:
        """
        Update the observed counts of each state symbol given a new observation.

        Args:
            obs (np.ndarray): discretized observations (batch_size, state_dim, discretize_dim)
        """
        obs = torch.tensor(obs.ravel(), dtype=torch.int64, device=self.device)
        self.matrix[self._batch_idx, self._state_idx, obs] += 1

    def __call__(self) -> np.ndarray:
        """
        Compute the information content across each batch element.

        Returns:
            ndarray of shape (batch_size, ) containing information content in bits

        Note: We assume independence across state axis.
        """
        f.normalize(self.matrix, p=1, dim=2, out=self.p_matrix)
        torch.log2(self.p_matrix+1e-6, out=self.logp_matrix)
        torch.mul(self.p_matrix, self.logp_matrix, out=self.p_matrix)
        return - torch.sum(self.p_matrix, dim=(2, 1)).cpu().numpy()


class NormDeltaInformation(Information):
    """
    Computes normalized change in information content.

    Args:
        batch_size (int): size of axis of independent trajectory samples
        state_dim (int): size of axis of state dimension
        discretize_dim (int): size of axis of discretized dimension
        device (torch.device): torch device on which to perform computation
        norm_factor (float): normalization factor
    """
    def __init__(
        self,
        batch_size: int,
        state_dim: int,
        discretize_dim: int,
        device: torch.device,
        norm_factor: float,
    ):
        super().__init__(batch_size, state_dim, discretize_dim, device)
        self.norm_factor = norm_factor
        self.last_entropy = None

    def __call__(self) -> np.ndarray:
        current_entropy = super().__call__()
        if self.last_entropy is None:
            result = self.norm_factor * current_entropy
        else:
            result = self.norm_factor * (current_entropy - self.last_entropy)
        self.last_entropy = current_entropy
        return result
