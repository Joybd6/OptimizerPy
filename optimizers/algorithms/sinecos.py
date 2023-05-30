from optimizers.algorithms import Algorithms
import numpy as np


class SineCos(Algorithms):
    r1 = None
    r2 = None
    r3 = None
    r4 = None
    a = None

    def __init__(self, dimension, max_iter, a=2):
        super().__init__(self.__class__.__name__, dimension, max_iter)
        self.a = a

    def step(self, iteration):
        if self._current_agent is None or self._local_optimum_agent is None or self._global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        if self.r4 < 0.5:
            updated_agent = self._current_agent + self.r1 * np.sin(self.r2) * np.abs(
                self.r3 * self._global_optimum_agent
                - self._current_agent)
        else:
            updated_agent = self._current_agent + self.r1 * np.cos(self.r2) * np.abs(
                self.r3 * self._global_optimum_agent
                - self._current_agent)
        return updated_agent

    def update_algorithm_state(self, iteration):
        self.r1 = self.a - self.a * (iteration / self.max_iter)
        self.r2 = 2 * np.pi * np.random.rand()
        self.r3 = np.random.rand()
        self.r4 = np.random.rand()
