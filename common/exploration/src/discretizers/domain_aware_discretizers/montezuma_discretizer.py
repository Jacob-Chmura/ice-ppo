import numpy as np
from .base_domain_aware_discretizer import DomainAwareDiscretizer

class MontezumaDiscretizer(DomainAwareDiscretizer):
    """
    Discretize according to checklist for the first level of Montezuma's Revenge
    """
    # Montezuma 64x64 key pixel location slices
    LEFT_FLOOR =         (slice(None), slice(45, 55), slice(10, 20))
    LEFT_LADDER =        (slice(None), slice(45, 55), slice(8, 11))
    LEFT_PLATFORM =      (slice(None), slice(30, 42), slice(4, 15))
    MIDDLE_LADDER =      (slice(None), slice(30, 40), slice(31, 33))
    RIGHT_FLOOR =        (slice(None), slice(45, 55), slice(40, 50))
    RIGHT_LADDER =       (slice(None), slice(45, 55), slice(53, 56))
    RIGHT_PLATFORM =     (slice(None), slice(30, 42), slice(50, 60))
    TOP_LEFT_PLATFORM =  (slice(None), slice(17, 27), slice(45, 65))
    TOP_PLATFORM =       (slice(None), slice(10, 15), slice(5, 60))
    TOP_RIGHT_PLATFORM = (slice(None), slice(17, 27), slice(0, 20))
    KEY =                (slice(None), slice(30, 34), slice(5, 8))

    # Average Pixel Intensity to mark the agent is present in frame
    AGENT_THRESHOLD = 30

    # Objective encodings
    AGENT_ON_MIDDLE_LADDER =      1
    AGENT_ON_RIGHT_PLATFORM =     2
    AGENT_ON_RIGHT_LADDER =       3
    AGENT_ON_RIGHT_FLOOR =        4
    AGENT_ON_LEFT_FLOOR =         5
    AGENT_ON_LEFT_LADDER =        6
    AGENT_ON_LEFT_PLATFORM =      7
    AGENT_ON_TOP_LEFT_PLATFORM =  8
    AGENT_ON_TOP_RIGHT_PLATFORM = 9
    AGENT_ON_TOP_PLATFORM =       10
    AGENT_ELSEWHERE =             0

    AGENT_WITHOUT_KEY = 1
    AGENT_WITH_KEY =    2

    def __init__(self):
        discretize_dim = 3 # (not complete, complete without key, complete with key)
        discretized_state_dim = 11 # (state is mapped according to above task list)
        super().__init__(discretize_dim, discretized_state_dim)

    def create_checklist(self, obs: np.ndarray):
        obs = (obs[:, :, :, 0] * 256).astype(int) # Extract R-channel
        task_positions = self._get_task_positions(obs)
        key_acquired = self._get_key_acquired(obs)
        checklist = task_positions * key_acquired[:, None]
        return checklist

    def _get_task_positions(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract current position of agent and output the task index of the position.

        Args:
            obs (ndarray): high dimensional continuous observation

        Returns:
            one-hot encoded array of size (batch_size, discretize_dim) s.t.
            ret_array[i, j] = 1 iff agent i is accomplishing objective j
        """
        current_task_positions = np.zeros(obs.shape[0], self.discretize_dim)
        current_task_positions[np.where(obs[self.MIDDLE_LADDER     ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_MIDDLE_LADDER
        current_task_positions[np.where(obs[self.RIGHT_PLATFORM    ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_RIGHT_PLATFORM
        current_task_positions[np.where(obs[self.RIGHT_LADDER      ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_RIGHT_LADDER
        current_task_positions[np.where(obs[self.RIGHT_FLOOR       ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_RIGHT_FLOOR
        current_task_positions[np.where(obs[self.LEFT_FLOOR        ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_LEFT_FLOOR
        current_task_positions[np.where(obs[self.LEFT_LADDER       ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_LEFT_LADDER
        current_task_positions[np.where(obs[self.LEFT_PLATFORM     ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_LEFT_LADDER
        current_task_positions[np.where(obs[self.TOP_LEFT_PLATFORM ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_TOP_LEFT_PLATFORM
        current_task_positions[np.where(obs[self.TOP_PLATFORM      ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_TOP_PLATFORM
        current_task_positions[np.where(obs[self.TOP_RIGHT_PLATFORM].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_TOP_RIGHT_PLATFORM
        current_task_positions[np.where(obs[self.MIDDLE_LADDER     ].mean(axis=(1, 2)) > self.AGENT_THRESHOLD)] = self.AGENT_ON_MIDDLE_LADDER
        return current_task_positions

    def _get_key_acquired(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns true iff the agent has acquired the key

        Args:
            obs (ndarray): high dimensional continuous observation

        Returns:
            array of size (batch_size,) encoding whether or not each agent acquired the key

        Note: once the key is acquired the pixels where the key was will be all black
            but this is not perfect since the agent's pixels could trigger false negatives.
        """
        key_acquired = np.full(shape=obs.shape[0], fill_value=self.AGENT_WITHOUT_KEY)
        key_acquired[np.where(np.all(obs[self.KEY] == 0, axis=(1, 2)))] = self.AGENT_WITH_KEY
        return key_acquired
