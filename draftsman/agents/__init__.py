from .dqn_agent import DQNAgent
from .random_agent import RandomAgent
from .rnd_agent import RNDAgent

agents = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "rnd": RNDAgent,
}
