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

# Import Optimizers
from optimizers import Optimizer


class HOptimizer(Optimizer):
    def __init__(self, objective_function, population, algorithm, callable_function):
        """

        :param objective_function: Python callable function
        :param population: Population object
        :param algorithm: HGSO algorithm should be in the first position
        :param callable_function: Algorithms callable function should in order with algorithms
        """
        super().__init__(objective_function, population, algorithm, callable_function)
        if not isinstance(algorithm[0], HGSO):
            raise TypeError("First algorithm must be a HGSO object")

    def run(self, max_iter):