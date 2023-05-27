from abc import ABC, abstractmethod
# from Optimizers.Population import Population
import numpy as np


class Algorithms(ABC):
    local_optimum_agent = None
    local_worst_agent = None
    global_optimum_agent = None
    current_agent = None

    def __init__(self, name, dimension, max_iter):
        self._name = name
        self.max_iter = max_iter
        self.dimension = dimension

    @property
    def name(self):
        return self._name

    @abstractmethod
    def step(self, iteration):
        """This method is called in each population step"""
        pass

    @abstractmethod
    def update_algorithm_state(self, iteration):
        """This method is called in each population step"""

    def set_agents(self, current_agent, local_optimum_agent=None, local_worst_agent=None,
                   global_optimum_agent=None):
        self.current_agent = current_agent
        self.local_optimum_agent = local_optimum_agent
        self.local_worst_agent = local_worst_agent
        self.global_optimum_agent = global_optimum_agent




class Jaya(Algorithms):

    def __init__(self, dimension, max_iter):
        super().__init__(self.__class__.__name__, dimension, max_iter)

    def step(self, iteration):
        if self.current_agent is None or self.local_optimum_agent is None or self.global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        r0 = np.random.rand(self.dimension)
        r1 = np.random.rand(self.dimension)
        updated_population = self.current_agent + r0 * (
                self.global_optimum_agent - np.abs(2 * self.current_agent - self.local_optimum_agent)) - r1 * (
                                     self.local_optimum_agent - self.current_agent)
        return updated_population

    def update_algorithm_state(self, iteration):
        pass


class Butterfly(Algorithms):
    _fragrance = None
    random_agent_j = None
    random_agent_k = None

    def __init__(self, name, dimension, max_iter, c=0.05, p=0.8):
        super().__init__(name, dimension, max_iter)
        self.c = c
        self.p = p
        self.a = 0.01

    def set_random_agents(self, random_agent_j, random_agent_k):
        self.random_agent_j = random_agent_j
        self.random_agent_k = random_agent_k

    @property
    def fragrance(self):
        return self._fragrance

    @fragrance.setter
    def fragrance(self, value):
        self._fragrance = self.c * np.power(value, self.a)

    def step(self, iteration):
        if self.current_agent is None or self.local_optimum_agent is None or self.global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        r = np.random.rand()
        if r < self.p:
            updated_agent = self.current_agent + (np.sqaure(r) * self.local_optimum_agent - self.current_agent) * \
                            self.fragrance
        else:
            updated_agent = self.current_agent + (np.square(r) * self.random_agent_j - self.random_agent_k) * \
                            self.fragrance
        return updated_agent

    def update_algorithm_state(self, iteration):
        pass


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
        if self.current_agent is None or self.local_optimum_agent is None or self.global_optimum_agent is None:
            raise ValueError("Population, local optimum and global optimum must be set before running the algorithm")

        if self.r4 < 0.5:
            updated_agent = self.current_agent + self.r1 * np.sin(self.r2) * np.abs(self.r3 * self.global_optimum_agent
                                                                                    - self.current_agent)
        else:
            updated_agent = self.current_agent + self.r1 * np.cos(self.r2) * np.abs(self.r3 * self.global_optimum_agent
                                                                                    - self.current_agent)
        return updated_agent

    def update_algorithm_state(self, iteration):
        r1 = self.a - self.a * (iteration / self.max_iter)
        r2 = 2 * np.pi * np.random.rand()
        r3 = np.random.rand()
        r4 = np.random.rand()