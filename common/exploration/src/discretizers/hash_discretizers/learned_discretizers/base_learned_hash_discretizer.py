from abc import abstractmethod
from collections import deque
import random
import numpy as np
import torch
import torch.optim as optim
from ..base_hash_discretizer import HashDiscretizer


class LearnedHashDiscretizer(HashDiscretizer):
    """
    Observation discretizer interface based on a learned hash function.

    Args:
        discretized_state_dim (int): the size of the (reduced) state dimension
        hash_dim (int): the dimension of the hash function output
        update_every (int): every update_every observation we update the learned model
        device (torch.device): the device to put model computation on
    """
    def __init__(
        self,
        discretized_state_dim: int,
        hash_dim: int,
        update_every: int,
        device=str,
    ):
        super().__init__(discretized_state_dim, hash_dim)
        self.update_every = update_every
        self.device = torch.device(device)
        self.count = 0
        self.batch_size = 32
        self.model = self._init_model().to(device=self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-4)
        self.buffer = deque(maxlen=1000)

    @abstractmethod
    def _init_model(self):
        """
        Initialize the model that will be used to extract discrete hash codes from state.
        """

    @abstractmethod
    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the input states and model produced outputs.

        Args:
            inputs (torch.Tensor): input data from batch
            outputs (torch.Tensor): model outputs from batch

        Returns:
            torch.FloatTensor: scalar loss
        """

    def _update(self):
        """
        Update the model.
        """
        inputs = np.vstack(random.choices(self.buffer, k=self.batch_size))
        inputs = inputs.transpose(0, 3, 1, 2)
        inputs = torch.Tensor(inputs).to(device=self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(inputs, outputs)
        loss.backward()
        self.optimizer.step()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.count += 1
        self.buffer.extend(np.split(obs, obs.shape[0]))
        if self.count % self.update_every == 0:
            self._update()
        return super().__call__(obs)
