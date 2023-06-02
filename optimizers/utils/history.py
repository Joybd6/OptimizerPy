import numpy as np


class History:
    _agents = []
    _best_agent = None

    def __init__(self):
        ...

    def add_agent(self, agent):
        if isinstance(agent, np.ndarray):
            self._agents.append(agent)

    def get(self, name):
        if name == 'agents':
            return self._agents
        elif name == 'best_agent':
            return self._best_agent
        else:
            raise ValueError("Invalid name")

    def set_best_agent(self, agent):
        if not isinstance(agent, (list, tuple)):
            raise TypeError("agent must be a list or tuple")
        self._best_agent = agent

