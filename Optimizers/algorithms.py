from abc import ABC, abstractmethod
# from Optimizers.Population import Population
import numpy as np


class Algorithms(ABC):
    _local_optimum_agent = None
    _local_worst_agent = None
    _global_optimum_agent = None
    _current_agent = None

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
        if current_agent.shape[0] != self.dimension:
            raise ValueError("current agent shape must be equal to (dimension,)")

        if local_optimum_agent is not None and local_optimum_agent.shape[0] != self.dimension:
            raise ValueError("local optimum agent shape must be equal to (dimension,)")

        if local_worst_agent is not None and local_worst_agent.shape[0] != self.dimension:
            raise ValueError("local worst agent shape must be equal to (dimension,)")

        if global_optimum_agent is not None and global_optimum_agent.shape[0] != self.dimension:
            raise ValueError("global optimum agent shape must be equal to (dimension,)")

        self._current_agent = current_agent
        self._local_optimum_agent = local_optimum_agent
        self._local_worst_agent = local_worst_agent
        self._global_optimum_agent = global_optimum_agent


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


class Butterfly(Algorithms):
    _fragrance = None
    _random_agent_j = None
    _random_agent_k = None

    def __init__(self, name, dimension, max_iter, c=0.05, p=0.8):
        super().__init__(name, dimension, max_iter)
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


# This is the implementation of HGSO algorithm's class. All the parameters and constants are set to the default values
class HGSO(Algorithms):
    _l1 = 5e-2
    _l2 = 100
    _l3 = 1e-2
    _t0 = 298.15
    _K = 1
    _epsilon = 0.05
    _c1 = 0.1
    _c2 = 0.2
    _best_agent_in_cluster = None
    _current_agent_fitness = None
    _best_local_agent_fitness = None
    _cluster_index = 0
    _agent_index = 0

    def __init__(self, dimension, max_iter, cluster_size, population_size_in_cluster, alpha=0.5, beta=0.5):
        super().__init__(self.__class__.__name__, dimension, max_iter)
        self.cluster_size = cluster_size
        self.population_size_in_cluster = population_size_in_cluster
        self.alpha = alpha
        self.beta = beta

        # Initializing the parameters of the algorithm
        self._H_j = self._l1 * np.random.rand(self.cluster_size)
        self._C_j = self._l3 * np.random.rand(self.cluster_size)
        self._P_ij = self._l2 * np.random.rand(self.population_size_in_cluster, self.cluster_size)
        self._S_ij = self._update_solubility()

    # Updating the solubility of the Gas
    def _update_solubility(self):
        return self._K * self._H_j * self._P_ij

    # Setters and getters of the parameters best_agent_in_cluster
    @property
    def best_agent_in_cluster(self):
        return self._best_agent_in_cluster

    @best_agent_in_cluster.setter
    def best_agent_in_cluster(self, value):
        if value.shape[0] != self.dimension:
            raise ValueError("best agent in cluster shape must be equal to (dimension,)")
        self._best_agent_in_cluster = value

    # Setters and getters of the parameters current_agent_fitness
    @property
    def current_agent_fitness(self):
        return self._current_agent_fitness

    @current_agent_fitness.setter
    def current_agent_fitness(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("current agent fitness must be a integer or float")
        self._current_agent_fitness = value

    # Setters and getters of the best_global_agent_fitness
    @property
    def best_local_agent_fitness(self):
        return self._best_local_agent_fitness

    @best_local_agent_fitness.setter
    def best_local_agent_fitness(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("best global agent fitness must be a integer or float")
        self._best_local_agent_fitness = value

    # Setters and getters of the parameters cluster_index
    @property
    def cluster_index(self):
        return self._cluster_index

    @cluster_index.setter
    def cluster_index(self, value):
        if not isinstance(value, int):
            raise TypeError("cluster index must be an integer")
        self._cluster_index = value

    # Setters and getters of the parameters agent_index
    @property
    def agent_index(self):
        return self._agent_index

    @agent_index.setter
    def agent_index(self, value):
        if not isinstance(value, int):
            raise TypeError("agent index must be an integer")
        self._agent_index = value

    def _check_parameters(self):
        if self._current_agent is None:
            raise ValueError("current agent must be set before running the algorithm")

        if self._local_optimum_agent is None:
            raise ValueError("local optimum agent must be set before running the algorithm")

        if self._global_optimum_agent is None:
            raise ValueError("global optimum agent must be set before running the algorithm")

        if self._best_agent_in_cluster is None:
            raise ValueError("best agent in cluster must be set before running the algorithm")

        if self._current_agent_fitness is None:
            raise ValueError("current agent fitness must be set before running the algorithm")

        if self._best_local_agent_fitness is None:
            raise ValueError("best local agent fitness must be set before running the algorithm")

    def step(self, iteration):
        self._check_parameters()

        gamma = self.beta * np.exp(-(self._best_local_agent_fitness + self._epsilon) / (self._current_agent_fitness +
                                                                                        self._epsilon))
        f = -1 if np.random.rand() < 0.5 else 1
        r = np.random.rand()
        updated_current_agent = self._current_agent + f * r * gamma * (
                self._best_agent_in_cluster - self._current_agent) + f * r * self.alpha * \
                                (self._S_ij[self.agent_index, self.cluster_index] *
                                 self._local_optimum_agent - self._current_agent)

        return updated_current_agent

    def update_algorithm_state(self, iteration):
        t = - (iteration + 1) / self.max_iter

        self._H_j = self._H_j * np.exp(-self._C_j * ((1 / t) - (1 / self._t0)))
        self._S_ij = self._update_solubility()


# Implementation of Owl search optimization's step
class OwlSearch(Algorithms):
    # Owl search parameters
    # I is the normalized fitness value of the current agent
    _normalized_fitness = None
    _EPSILON = 0.00000001

    def __init__(self, dimension, max_iter):
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
