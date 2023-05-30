from optimizers.algorithms import Algorithms
import numpy as np


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
