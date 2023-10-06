from .base_discretizer import Discretizer
from .uniform_discretizer import UniformDiscretizer
from .hash_discretizers.base_hash_discretizer import HashDiscretizer
from .hash_discretizers.bass_discretizer import BassDiscretizer
from .hash_discretizers.learned_discretizers.base_learned_hash_discretizer import LearnedHashDiscretizer
from .hash_discretizers.learned_discretizers.ae_discretizer import AEDiscretizer
from .hash_discretizers.learned_discretizers.vqvae_discretizer import VQVAEDiscretizer
from .domain_aware_discretizers.montezuma_discretizer import MontezumaDiscretizer


def discretizer_from_config(discretizer_type: str, discretizer_config: dict):
    """
    Initialize discretizer from config.
    """
    if discretizer_type == "UniformDiscretizer":
        discretizer = UniformDiscretizer(
            discretize_dim=discretizer_config.get("discretizer_dim", 256),
            discretized_state_dim=discretizer_config.get("discretized_state_dim", 3*64*64),
        )
    elif discretizer_type == "BassDiscretizer":
        discretizer = BassDiscretizer(
            discretized_state_dim=discretizer_config.get("discretized_state_dim", 32),
            original_state_dim=discretizer_config.get("original_state_dim", 3*64*64),
            cell_size=discretizer_config.get("cell_size", 20),
            num_bins=discretizer_config.get("num_bins", 20),
        )
    elif discretizer_type == "AEDiscretizer":
        discretizer = AEDiscretizer(
            discretized_state_dim=discretizer_config.get("discretized_state_dim", 32),
            hash_dim=discretizer_config.get("hash_dim", 128),
            update_every=discretizer_config.get("update_every", 3),
            device=discretizer_config.get("device", "cuda"),
            noise_range=discretizer_config.get("noise_range", 0.3),
            lmbda=discretizer_config.get("lmbda", 10),
        )
    elif discretizer_type == "MontezumaDiscretizer":
        discretizer = MontezumaDiscretizer()
    else: 
        raise ValueError(f"Unknown discretizer: {discretizer_type}")
    return discretizer
