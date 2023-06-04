from optimizers.algorithms import Algorithms
import numpy as np


class SineCos(Algorithms):
    """
    Sine Cosine Optimization Algorithm
    """

    r1 = None
    r2 = None
    r3 = None
    r4 = None
    a = None

    def __init__(self, dimension, **kwargs):
        """

        :param dimension:
        :param max_iter:
        :param kwargs:
        """
        super().__init__(self.__class__.__name__, dimension)
        self.a = kwargs['a']
        self.update_algorithm_state(0, 1)

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

    def update_algorithm_state(self, iteration, max_iter):
        #print(self.max_iter)
        self.r1 = self.a - self.a * ((iteration+1) / max_iter)
        self.r2 = 2 * np.pi * np.random.rand()
        self.r3 = np.random.rand()
        self.r4 = np.random.rand()


def sinecos_callable(algorithm: Algorithms, population, current_id, iteration):
    algorithm.current_agent = population.population[current_id]
    algorithm.local_optimum_agent = population.local_optimum[0]
    algorithm.global_optimum_agent = population.global_optimum[0]
