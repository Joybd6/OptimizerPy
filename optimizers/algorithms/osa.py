from optimizers.algorithms import Algorithms
import numpy as np


class OwlSearch(Algorithms):
    # Owl search parameters
    # I is the normalized fitness value of the current agent
    _normalized_fitness = None
    _EPSILON = 0.00000001

    def __init__(self, dimension, max_iter, **kwargs):
        super().__init__(self.__class__.__name__, dimension, max_iter)

    @property
    def normalized_fitness(self):
        return self._normalized_fitness

    @normalized_fitness.setter
    def normalized_fitness(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("I must be a integer or float")
        self._normalized_fitness = value

    def step(self, iteration):
        if self._current_agent is None:
            raise ValueError("current agent must be set before running the algorithm")

        if self._global_optimum_agent is None:
            raise ValueError("global optimum agent must be set before running the algorithm")

        if self._normalized_fitness is None:
            raise ValueError("The current normalized fitness value must be set before running the algorithm")

        # Ri is the Euclidean distance between the current agent and the global optimum agent
        Ri = np.linalg.norm(self._current_agent - self._global_optimum_agent)
        # Ici is the intensity change of the current agent
        Ici = (self._normalized_fitness / (Ri ** 2 + self._EPSILON)) + np.random.rand()
        pvm = np.random.rand()
        alpha = np.random.uniform(0, 0.5)
        beta = 1.9 * ((iteration + 1) / self.max_iter)

        if pvm < 0.5:
            updated_agent = self._current_agent + beta * Ici * np.abs(alpha * self._global_optimum_agent -
                                                                      self._current_agent)
        else:
            updated_agent = self._current_agent - beta * Ici * np.abs(alpha * self._global_optimum_agent -
                                                                      self._current_agent)

        return updated_agent

    def update_algorithm_state(self, iteration):
        pass
