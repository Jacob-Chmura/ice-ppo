import numpy as np
import torch
import torch.nn as nn
from .base_learned_hash_discretizer import LearnedHashDiscretizer
from .models import VQVAE


class VQVAEDiscretizer(LearnedHashDiscretizer):
    """
    Observation discretizer interface based on a learned vq-vae latent codebook.

    Args:
        discretized_state_dim (int): the size of the (reduced) state dimension
        hash_dim (int): the dimension of the hash function output
        update_every (int): every update_every observation we update the learned model
        device (torch.device): the device to put model computation on
    """
    def __init__(
        self,
        discretize_dim: int,
        hash_dim: int,
        update_every: int,
        device: torch.device,
        commitment_cost: float,
    ):
        self.commitment_cost = commitment_cost
        super().__init__(discretize_dim, hash_dim, update_every, device)
        self.reconstruction_loss = nn.MSELoss()

    def _init_model(self):
        return VQVAE(
            num_hiddens=96,
            num_embeddings=self.hash_dim,
            embedding_dim=96,
            commitment_cost=self.commitment_cost,
        )

    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        VQ-VAE loss consists of likelihood term, an embedding error term, and a commitment loss.
        """
        recon_loss = self.reconstruction_loss(inputs, outputs)
        vq_loss = self.model.vq_loss
        return recon_loss + vq_loss

    def hash(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.transpose(0, 3, 1, 2).astype(np.float32)
        obs = torch.from_numpy(obs).to(device=self.device)
        with torch.no_grad():
            _, _, obs = self.model.encode(obs)
            obs = obs.cpu().numpy()
        return obs
