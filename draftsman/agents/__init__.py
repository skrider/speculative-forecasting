from .dqn_agent import DQNAgent
from .random_agent import RandomAgent
from .rnd_agent import RNDAgent
from .deterministic_agent import DeterministicAgent

agents = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "rnd": RNDAgent,
    "deterministic": DeterministicAgent,
}
