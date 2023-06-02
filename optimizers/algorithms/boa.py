from optimizers.algorithms import Algorithms
import numpy as np


class Butterfly(Algorithms):
    """
    Butterfly Optimization Algorithm
    :param dimension: dimension of the population
    :param max_iter: maximum number of iterations
    :param c: c value for calculating fragrance
    :param p: p value for calculating fragrance
    """
    _fragrance = None
    _random_agent_j = None
    _random_agent_k = None

    def __init__(self, dimension, **kwargs):

        """
        Butterfly Optimization Algorithm
        :param dimension: dimension of the population
        :param max_iter: maximum number of iterations
        :param c: c value for calculating fragrance
        :param p: p value for calculating fragrance
        """
        super().__init__(self.__class__.__name__, dimension)
        self.c = kwargs['c']
        self.p = kwargs['p']
        self.a = 0.01

    @property
    def random_agent_j(self):
        return self._random_agent_j

    @random_agent_j.setter
    def random_agent_j(self, value):
        if value.shape[0] != self.dimension:
            raise ValueError("random agent j shape must be equal to (dimension,)")
        self._random_agent_j = value

    @property
    def random_agent_k(self):
        return self._random_agent_k

    @random_agent_k.setter
    def random_agent_k(self, value):
        if value.shape[0] != self.dimension:
            raise ValueError("random agent k shape must be equal to (dimension,)")
        self._random_agent_k = value

    @property
    def fragrance(self):
        return self._fragrance

    @fragrance.setter
    def fragrance(self, value):
        self._fragrance = self.c * np.power(value, self.a)

    def step(self, iteration):
        """
        This method is called in each population step
        :param: iteration:
        :return: updated agent:

        """
        if self._current_agent is None or self._local_optimum_agent is None or self._global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        r = np.random.rand()
        if r < self.p:
            updated_agent = self._current_agent + (np.square(r) * self._local_optimum_agent - self._current_agent) * \
                            self.fragrance
        else:
            updated_agent = self._current_agent + (np.square(r) * self._random_agent_j - self._random_agent_k) * \
                            self.fragrance
        return updated_agent

    def update_algorithm_state(self, iteration):
        pass


def butterfly_callable(algorithm: Algorithms, population, current_id, iteration):
    algorithm.fragrance = population.eval_value[current_id]
    algorithm.random_agent_j = population.population[np.random.randint(0, population.population.shape[0])]
    algorithm.random_agent_k = population.population[np.random.randint(0, population.population.shape[0])]
    algorithm.current_agent = population.population[current_id]
    algorithm.global_optimum_agent = population.global_optimum[0]
    algorithm.local_optimum_agent = population.local_optimum[0]
