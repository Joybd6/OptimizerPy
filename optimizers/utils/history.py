import numpy as np
import matplotlib.pyplot as plt

class History:
    name = None
    _agents = None
    _best_agent = None
    _fitness = None
    def __init__(self):
        self._agents = []
        self._fitness = []

    def add_agent(self, agent):
        if not isinstance(agent, (list, tuple)):
            raise TypeError("agent must be a list or tuple")
        self._agents.append(agent[0])
        self._fitness.append(agent[1])

    def get(self, name):
        if name == 'agents':
            return self._agents
        elif name == 'best_agent':
            return self._best_agent
        elif name == 'fitness':
            return self._fitness
        else:
            raise ValueError("Invalid name")

    def set_best_agent(self, agent):
        if not isinstance(agent, (list, tuple)):
            raise TypeError("agent must be a list or tuple")
        self._best_agent = agent

    def plot(self):
        #if self.name is not None:
            #plt.label(self.name)
        plt.plot(self._fitness, label=self.name)
        #plt.xlabel("Iteration")
        #plt.ylabel("Optimal Value")

