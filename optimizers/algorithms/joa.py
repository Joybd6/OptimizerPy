from optimizers.algorithms import Algorithms
import numpy as np


class Jaya(Algorithms):

    def __init__(self, dimension, **kwargs):
        super().__init__(self.__class__.__name__, dimension,)

    def step(self, iteration):
        if self._current_agent is None or self._local_optimum_agent is None or self._local_worst_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        r0 = np.random.rand(self.dimension)
        r1 = np.random.rand(self.dimension)

        updated_population = self._current_agent + r0 * (
                self._local_optimum_agent - np.abs(self._current_agent)) - r1 * (
                                     self._local_worst_agent - np.abs(self._current_agent))
        return updated_population

    def update_algorithm_state(self, iteration):
        pass


def jaya_callable(algorithm: Algorithms, population, current_id, iteration):

    worst_agent = population.population[np.argmin(population.eval_value)] if population.optimization == 'max' \
        else population.population[np.argmax(population.eval_value)]
    algorithm.current_agent = population.population[current_id]
    algorithm.local_optimum_agent = population.local_optimum[0]
    algorithm.local_worst_agent = worst_agent

