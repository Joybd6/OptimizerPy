from optimizers.population import Population
from optimizers.algorithms import Algorithms
from optimizers.utils import History
from copy import deepcopy
import numpy as np
import tqdm

class Optimizer:
    """
    This class is the main class to hold all the building blocks of the optimization process.
    """

    _global_best_agent = None
    _local_optimum_agent = None

    def __init__(self, objective_function: callable, population: Population, algorithms, algorithms_callback):
        """
        :param objective_function:
        :param population: Population object
        :param algorithms: list or tuple of Algorithms objects or a single Algorithms object. For multiple algorithms the run method will need to be override
        :param algorithms_callback: list or tuple of callable objects or a single callable object. Should be in order with algorithms
        """

        if not callable(objective_function):
            raise TypeError("objective_function must be callable")

        # Check if algorithms_callback is a list or tuple of callable objects or a single callable object
        if callable(algorithms_callback):
            algorithms_callback = [algorithms_callback]
        elif isinstance(algorithms_callback, (list, tuple)):
            for callback in algorithms_callback:
                if not callable(callback):
                    raise TypeError("algorithms_callback must be a list or tuple of callable objects")
        else:
            raise TypeError("algorithms_callback must be a list or tuple of callable objects")

        # Check if algorithms is a list or tuple of Algorithms objects or a single Algorithms object
        if isinstance(algorithms, Algorithms):
            algorithms = [algorithms]
        elif isinstance(algorithms, (list, tuple)):
            for algo in algorithms:
                if not isinstance(algo, Algorithms):
                    raise TypeError("algorithms must be a list or tuple of Algorithms objects")
        else:
            raise TypeError("algorithms must be a list or tuple of Algorithms objects")

        # Assign the values
        self._obj = objective_function
        self._population = population
        self._algorithms = algorithms
        self._algorithms_callback = algorithms_callback
        self.history = History()

    @property
    def population(self):
        return self._population

    def run(self, max_iter):
        """
        This method runs the optimization process.
        :return:
        """
        self._algorithms[0].max_iter = max_iter
        self.population.update_global_optimum()

        tq = tqdm.tqdm(range(max_iter), desc="Optimizing", unit="iter", ncols=100, ascii=" #",
                       postfix={"fitness": self.population.global_optimum[1]})
        for iteration in tq:
            temp_pop = []
            for agent_id, agent in enumerate(self._population.population):
                self._algorithms_callback[0](self._algorithms[0], deepcopy(self._population), agent_id, iteration)
                temp_pop.append(self._algorithms[0].step(iteration))

            self._population.population = np.array(temp_pop)
            self._population.update_global_optimum()

            for call in self._algorithms[0].per_iter_callback:
                call(self._population, iteration, max_iter)

            self._algorithms[0].update_algorithm_state(iteration, max_iter)
            tq.set_postfix({"fitness": self.population.global_optimum[1]})
