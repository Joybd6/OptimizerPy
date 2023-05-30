from optimizers.algorithms import Algorithms
import numpy as np


class Butterfly(Algorithms):
    """
    Butterfly Optimization Algorithm
    :Constructor arguments:
        :param dimension:
        :param max_iter:
        :param c:
        :param p:
    """
    _fragrance = None
    _random_agent_j = None
    _random_agent_k = None

    def __init__(self, dimension, max_iter, c=0.05, p=0.8):
        """

        :param dimension:
        :param max_iter:
        :param c:
        :param p:
        """
        super().__init__(self.__class__.__name__, dimension, max_iter)
        self.c = c
        self.p = p
        self.a = 0.01

    def set_random_agents(self, random_agent_j, random_agent_k):

        if random_agent_j.shape[0] != self.dimension:
            raise ValueError("random agent j shape must be equal to (dimension,)")

        if random_agent_k.shape[0] != self.dimension:
            raise ValueError("random agent k shape must be equal to (dimension,)")

        self._random_agent_j = random_agent_j
        self._random_agent_k = random_agent_k

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
            updated_agent = self._current_agent + (np.sqaure(r) * self._local_optimum_agent - self._current_agent) * \
                            self.fragrance
        else:
            updated_agent = self._current_agent + (np.square(r) * self._random_agent_j - self._random_agent_k) * \
                            self.fragrance
        return updated_agent

    def update_algorithm_state(self, iteration):
        pass
