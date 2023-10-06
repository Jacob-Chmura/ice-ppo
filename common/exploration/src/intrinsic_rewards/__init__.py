from .ice import ICE
from .mice import MICE

def intrinsic_reward_from_config(discretizer, method_type: str, method_config: dict):
    """
    Initialize the exploration bonus method from config.
    """
    if method_type == "ICE":
        exploration = ICE(
            discretizer=discretizer,
            beta=method_config.get("beta_entropy", 0.01),
        )
    elif method_type == "MICE":
        exploration = MICE(
            discretizer=discretizer,
            beta_entropy=method_config.get("beta_entropy", 0.01),
            beta_mutual=method_config.get("beta_mutual", 0.01),
        )
    else:
        raise ValueError(f"Unknown method: {method_type}")
    return exploration
