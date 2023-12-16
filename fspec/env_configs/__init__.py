from .dqn_config import basic_dqn_config
from .random_agent_config import random_agent_config
from .rnd_config import rnd_config
from .deterministic_agent_config import deterministic_agent_config
from .mlp_config import basic_mlp_config

configs = {
    "dqn": basic_dqn_config,
    "random": random_agent_config,
    "rnd": rnd_config,
    "deterministic": deterministic_agent_config,
    "mlp": basic_mlp_config
}
