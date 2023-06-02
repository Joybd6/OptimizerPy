from optimizers.algorithms import Butterfly
from optimizers.algorithms import Jaya
from optimizers.population import Population
from optimizers.algorithms import butterfly_callable
from optimizers.algorithms import jaya_callable
from optimizers.algorithms import sinecos_callable
from optimizers.algorithms import SineCos
from optimizers import Optimizer
from optimizers.algorithms import OwlSearch
from optimizers.algorithms import owl_search_callable
import numpy as np
from optimizers.algorithms import HGSO
from optimizers.algorithms import hgso_callable

if __name__ == "__main__":
    def sphere(x):
        return np.sum(np.power(x, 2))


    dimension = 30
    upper_bound = np.array([100]).repeat(dimension)
    lower_bound = np.array([-100]).repeat(dimension)
    pop_size = 100

    # Population initialization
    population = Population(lower_bound, upper_bound, size=pop_size, dimension=dimension, objective_function=sphere)
    population.initialize(initializer="uniform", lower_bound=0, upper_bound=1)

    # Optimizer initialization
    # Test Butterfly
    butterfly_params = {'c': 0.2, 'p': 0.8}
    algorithm = Jaya(dimension=dimension, **{})
    sinecos_params = {'a': 1}
    #algorithm = OwlSearch(dimension=dimension)
    # Optimizer initialization

    # HGSO
    hgso_params = {'n_cluster': 5, "cluster_size": 20, "alpha": 0.8, "beta": 0.5}
    #algorithm = HGSO(dimension=dimension, **hgso_params)
    opt = Optimizer(sphere, population, algorithm, jaya_callable)
    opt.run(100)
    print(opt.population.global_optimum)
