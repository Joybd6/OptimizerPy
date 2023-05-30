from optimizers.algorithms import Algorithms
import numpy as np


class Jaya(Algorithms):

    def __init__(self, dimension, max_iter):
        super().__init__(self.__class__.__name__, dimension, max_iter)

    def step(self, iteration):
        if self._current_agent is None or self._local_optimum_agent is None or self._global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        r0 = np.random.rand(self.dimension)
        r1 = np.random.rand(self.dimension)

        updated_population = self._current_agent + r0 * (
                self._global_optimum_agent - np.abs(self._current_agent - self._local_optimum_agent)) - r1 * (
                                     self._local_optimum_agent - self._current_agent)
        return updated_population

    def update_algorithm_state(self, iteration):
        pass
