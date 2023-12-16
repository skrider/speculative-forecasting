from typing import Tuple
import numpy as np


class DeterministicAgent:
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int, action: int):
        super().__init__()
        self.num_actions = num_actions
        self.action = action

    def get_action(self, *args, **kwargs):
        return self.action

    def update(self, *args, **kwargs):
        return {}
