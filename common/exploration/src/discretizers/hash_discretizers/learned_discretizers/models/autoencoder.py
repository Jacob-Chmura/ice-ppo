import torch
import torch.nn as nn
from .common import Encoder
from .common import Decoder

class ConvAutoEncoder(nn.Module):
    """
    Convolution AutoEncoder with binary sigmoid layer.

    Args:
        num_hiddens: the hidden dimension for the encoder
        embedding_dim: the dimension of the latent space
    """
    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
    ):
        super().__init__()
        self._encoder = Encoder(3, num_hiddens)
        self._decoder = Decoder(embedding_dim, num_hiddens)
        self._pre_binarizer_linear = nn.Linear(num_hiddens, embedding_dim)
        self.latent_activations = None

    def forward(self, x):
        self.latent_activations = self.encode(x)
        x_recon = self._decoder(self.latent_activations)
        return x_recon

    def encode(self, x):
        z = self._encoder(x)
        z = z.squeeze()
        z = self._pre_binarizer_linear(z)
        z = torch.sigmoid(z)
        return z
