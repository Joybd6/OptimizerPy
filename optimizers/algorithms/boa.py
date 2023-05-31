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

    def __init__(self, dimension, max_iter, c=0.05, p=0.8, **kwargs):
        """

        :param dimension:
        :param max_iter:
        :param c(kwargs):
        :param p(kwargs):
        """
        super().__init__(self.__class__.__name__, dimension, max_iter)
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
            updated_agent = self._current_agent + (np.sqaure(r) * self._local_optimum_agent - self._current_agent) * \
                            self.fragrance
        else:
            updated_agent = self._current_agent + (np.square(r) * self._random_agent_j - self._random_agent_k) * \
                            self.fragrance
        return updated_agent

    def update_algorithm_state(self, iteration):
        pass
