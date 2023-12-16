from .dqn_agent import DQNAgent
from .random_agent import RandomAgent
from .rnd_agent import RNDAgent
from .deterministic_agent import DeterministicAgent
from .mlp_agent import MLPAgent


agents = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "rnd": RNDAgent,
    "deterministic": DeterministicAgent,
    "mlp": MLPAgent
}
