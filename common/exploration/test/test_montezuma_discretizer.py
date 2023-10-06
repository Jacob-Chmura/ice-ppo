import unittest
from exploration.src.discretizers import MontezumaDiscretizer

class MontezumaDiscretizerTest(unittest.TestCase):
    def setUp(self):
        self.discretizer = MontezumaDiscretizer()

    def _gen_obs(self):
        obs = None
        return obs

    def test_init(self):
        assert self.discretizer.discretize_dim == 3
        assert self.discretizer.discretized_state_dim == 11
