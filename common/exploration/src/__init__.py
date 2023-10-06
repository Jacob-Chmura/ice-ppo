from .intrinsic_rewards import intrinsic_reward_from_config
from .discretizers import discretizer_from_config

def exploration_from_config(exploration_config: dict):
    """
    Initialize the intrinsic exploration bonus based on experiment configs.
    """
    discretizer_type, method_type = exploration_config["discretizer"], exploration_config["method"]
    discretizer_config = exploration_config["discretizer_config"][discretizer_type]
    method_config = exploration_config["method_config"][method_type]
    discretizer = discretizer_from_config(discretizer_type, discretizer_config)
    return intrinsic_reward_from_config(discretizer, method_type, method_config)
