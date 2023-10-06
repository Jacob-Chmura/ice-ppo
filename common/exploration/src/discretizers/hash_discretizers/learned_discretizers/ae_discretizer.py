import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from .base_learned_hash_discretizer import LearnedHashDiscretizer
from .models import ConvAutoEncoder


class AEDiscretizer(LearnedHashDiscretizer):
    """
    Observation discretizer interface based on a learned autoencoder latent state.

    Args:
        discretized_state_dim (int): the size of the (reduced) state dimension
        hash_dim (int): the dimension of the hash function output
        update_every (int): every update_every observation we update the learned model
        device (torch.device): the device to put model computation on
        noise_range (float): range of uniform noise to add to latent state sigmoid output
                                to ensure saturated outputs.
        lmbda (float): auxilary loss coefficient pressures code layer to take on binary values
    """
    def __init__(
        self,
        discretized_state_dim: int,
        hash_dim: int,
        update_every: int,
        device: torch.device,
        noise_range: float,
        lmbda: float,
    ):
        super().__init__(discretized_state_dim, hash_dim, update_every, device)
        self.noise_distribution = Uniform(-noise_range, noise_range)
        self.lmbda = lmbda
        self.reconstruction_loss = nn.MSELoss()

    def _init_model(self):
        return ConvAutoEncoder(
            num_hiddens=96,
            embedding_dim=self.hash_dim,
        )

    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Objective function consists of a reconstruction loss
        and a term that pressures the binary code layer to take on binary values.
        """
        recon_loss = self.reconstruction_loss(inputs, outputs)
        latent_activations = self.model.latent_activations
        auxilary_loss = torch.mean(torch.minimum(
            torch.square(1 - latent_activations),
            torch.square(latent_activations)
        ))
        return recon_loss + self.lmbda * auxilary_loss

    def hash(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.transpose(0, 3, 1, 2).astype(np.float32)
        obs = torch.from_numpy(obs).to(device=self.device)
        with torch.no_grad():
            obs = self.model.encode(obs).cpu()
        noise = self.noise_distribution.sample(obs.shape)
        hashed_obs = torch.floor(obs + noise).cpu().numpy()
        return hashed_obs
