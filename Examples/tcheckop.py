import numpy as np
from copy import deepcopy
from optimizers.population import Population
# Import HGSO
from optimizers.algorithms import HGSO
from optimizers.algorithms import hgso_callable
# Import SinCos
from optimizers.algorithms import SineCos
from optimizers.algorithms import sinecos_callable
# Import Jaya
from optimizers.algorithms import Jaya
from optimizers.algorithms import jaya_callable
# Import Butterfly
from optimizers.algorithms import Butterfly
from optimizers.algorithms import butterfly_callable
# Import OwlSearch
from optimizers.algorithms import OwlSearch
from optimizers.algorithms import owl_search_callable
import tqdm

# Import Optimizers
from optimizers import Optimizer


class HOptimizer(Optimizer):
    _pmin = 0.2
    _pmax = 0.8
    _window = []
    _window_size = 0

    def __init__(self, objective_function, population, algorithm, callable_function, window_size=3):
        """

        :param objective_function: Python callable function
        :param population: Population object
        :param algorithm: HGSO algorithm should be in the first position
        :param callable_function: Algorithms callable function should in order with algorithms
        """
        super().__init__(objective_function, population, algorithm, callable_function)
        if not isinstance(algorithm[0], HGSO):
            raise TypeError("First algorithm must be a HGSO object")
        self._window_size = window_size

    def _check_updates(self):
        if self.population.optimization == 'min':
            if self.population.global_optimum[1] < self.population.local_optimum[1]:
                return 1
            else:
                return 0
        elif self.population.optimization == 'max':
            if self.population.global_optimum[1] > self.population.local_optimum[1]:
                return 1
            else:
                return 0

    def _shuffle_algorithm(self, algorithm_indexes, iteration, max_iter):
        p = self._pmax + (iteration / max_iter) * (self._pmin - self._pmax)

        if np.random.uniform(0, 1) < p:
            #print("Shuffling algorithms")
            np.random.shuffle(algorithm_indexes)
        return algorithm_indexes

    def run(self, max_iter):

        # Set the max iteration value for all algorithms
        for algorithm in self._algorithms:
            algorithm.max_iter = max_iter

        # Update the population
        self.population.update_global_optimum()

        idx = None  # Index of the algorithm

        # set the algorithm indexes for dynamically mapping to the cluster
        algorithm_indexes = np.arange(len(self._algorithms))

        # Set the current algorithm and callable function
        current_algorithm = None
        current_callable = None

        # Set the cluster number and cluster size
        n_cluster = self._algorithms[0].n_cluster
        cluster_size = self._algorithms[0].cluster_size

        # Initialize the iteration
        for i in range(max_iter):
            idx = -1
            temp_pop = []  # Temporary updated population

            # Iterate over the population
            for agent_id, agent in tqdm(enumerate(self._population.population)):

                # Update the current algorithm
                if int(agent_id % cluster_size) == 0:
                    idx = int((idx + 1) % len(self._algorithms))
                    current_algorithm = self._algorithms[algorithm_indexes[idx]]
                    current_callable = self._algorithms_callback[algorithm_indexes[idx]]

                # Update the agent
                current_callable(current_algorithm, self._population, agent_id, i)
                updated_agent = current_algorithm.step(i)
                temp_pop.append(updated_agent)

            # Update the population and global optimum
            self.population.population = np.array(temp_pop)
            self.population.update_global_optimum()

            for call in self._algorithms[0].per_iter_callback:
                call(deepcopy(self._population), i, max_iter)

            if self._check_updates() == 0:
                algorithm_indexes = self._shuffle_algorithm(algorithm_indexes, i, max_iter)
