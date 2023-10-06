import numpy as np
from .base_discretizer import Discretizer


class UniformDiscretizer(Discretizer):
    """
    Discretizes observation by uniformly bucketing into self.discretize_dim bins.
    """
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return (obs * self.discretize_dim).astype(int)
