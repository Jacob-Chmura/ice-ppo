import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Encoder
from .common import Decoder

class VQVAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder

    Args:
        num_hiddens: the hidden dimension for the encoder
        num_embeddings: the number of codebook vectors
        embedding_dim: the dimension of each codebook vector
        commitment_cost: the commitment cost loss coefficient
    """
    def __init__(
        self,
        num_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
    ):
        super().__init__()
        self._encoder = Encoder(3, num_hiddens)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens)

        self.vq_loss = None
        self.encodings = None

    def forward(self, x):
        self.vq_loss, z, self.encodings = self.encode(x)
        x_recon = self._decoder(z)
        return x_recon

    def encode(self, x):
        z = self._encoder(x)
        vq_loss, quantization, encodings = self._vq_vae(z)
        return vq_loss, quantization, encodings


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(self._embedding.weight**2, dim=1) - 2 *
            torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings
