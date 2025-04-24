import torch

class Agent:
    def __init__(self, config):
        self.config = config
        self.policy = lambda _state: torch.ones(3)

    def get_action(self, state):
        return self.policy(state)
