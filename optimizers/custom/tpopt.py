import tqdm
import numpy as np
from copy import deepcopy
from optimizers.algorithms import HGSO
# Import Optimizers
from optimizers import Optimizer


class TOptimizer(Optimizer):
    _pmin = 0.2
    _pmax = 0.8
    _window = []
    _window_size = 0
    _tp_threshold = None

    def __init__(self, objective_function, population, algorithm, callable_function, window_size=3, tp_threshold=10):
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
        self._tp_threshold = tp_threshold

    # Implementation of the triple check mechanism
    def __triple_check(self, iteration, max_iter):

        if len(self._window) < self._window_size:
            return self.population

        if len(self._window) == self._window_size:
            self._window.pop(0)

        # Appending the current population to the window
        self._window.append(deepcopy(self.population))

        # Check if the global optimum difference between two epochs
        # is greater than the threshold, if yes, it won't perform updates otherwise
        # it will perform the update using all algorithms and return the best one

        dif = None
        if self.population.optimization == 'max':
            dif = self._window[-1].global_optimum[1] - self._window[-2].global_optimum[1]
        elif self.population.optimization == 'min':
            dif = self._window[-2].global_optimum[1] - self._window[-1].global_optimum[1]

        threshold = self._tp_threshold * (iteration / max_iter)
        if dif >= threshold:
            return self.population

       # print("Performing triple check")

        updated_pop = self._window[-1]

        for pop_index in range(len(self._window) - 1):

            updated_pop_algorithm = self._window[pop_index]

            for a_index, algorithm in enumerate(self._algorithms):
                temp_pop = self._perform_tp_update(deepcopy(self._window[pop_index]), a_index, iteration)

                if self.population.optimization == 'min':
                    if temp_pop.global_optimum[1] < updated_pop.global_optimum[1]:
                        updated_pop = temp_pop
                elif self.population.optimization == 'max':
                    if temp_pop.global_optimum[1] > updated_pop.global_optimum[1]:
                        updated_pop_algorithm = temp_pop

            if self.population.optimization == 'min':
                if updated_pop_algorithm.global_optimum[1] < updated_pop.global_optimum[1]:
                    updated_pop = updated_pop_algorithm
            elif self.population.optimization == 'max':
                if updated_pop_algorithm.global_optimum[1] > updated_pop.global_optimum[1]:
                    updated_pop = updated_pop_algorithm

        return updated_pop

    def _perform_tp_update(self, population, algorithm_index, iteration):

        temp_pop = []
        for agent_id, agent in enumerate(population.population):
            self._algorithms_callback[algorithm_index](self._algorithms[algorithm_index], deepcopy(population),
                                                       agent_id, iteration)
            temp_pop.append(self._algorithms[algorithm_index].step(iteration))
            """
            Implement Rest of the updates
            """
        population.population = np.array(temp_pop)
        return population  # Return the updated population

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
            # print("Shuffling algorithms")
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

        #tq = tqdm.tqdm(range(max_iter), desc="Optimizing", unit="iter", ncols=100, ascii=" #",
                       #postfix={"fitness": self.population.global_optimum[1]})
        # Initialize the iteration
        for i in range(max_iter):
            idx = -1
            temp_pop = []  # Temporary updated population

            # Iterate over the population
            for agent_id, agent in enumerate(self._population.population):

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

            # Call the algorithms per iteration callback. In this custom optimizer the callback
            # is called only for the first algorithm
            for call in self._algorithms[0].per_iter_callback:
                call(deepcopy(self._population), i, max_iter)

            if self._check_updates() == 0:
                algorithm_indexes = self._shuffle_algorithm(algorithm_indexes, i, max_iter)

            if len(self._window) == self._window_size:
                self._window.pop(0)

            self._window.append(deepcopy(self.population))

            self._population = self.__triple_check(i, max_iter)
            self._window.pop(-1)
            self._window.append(deepcopy(self._population))

            #tq.set_postfix({"fitness": self.population.global_optimum[1]})
            print("Iteration: ", i, "Fitness: ", self.population.global_optimum[1])
