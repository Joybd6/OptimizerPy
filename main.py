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
import matplotlib.pyplot as plt
from benchmarking.benchmark import Rastrigin
from benchmarking.benchmark import Ackley, Quartic_with_noise
if __name__ == "__main__":

    obj = Quartic_with_noise()


    dimension = 30
    upper_bound = np.array([100]).repeat(dimension)
    lower_bound = np.array([-100]).repeat(dimension)
    pop_size = 100

    # Population initialization
    population = Population(lower_bound, upper_bound, size=pop_size, dimension=dimension, objective_function=obj.get_algorithm(), optimization="min")
    population.initialize(initializer="uniform", lower_bound=lower_bound, upper_bound=upper_bound)

    # Optimizer initialization
    # Test Butterfly
    butterfly_params = {'c': 0.2, 'p': 0.8}
    #algorithm = Jaya(dimension=dimension, **{})
    sinecos_params = {'a': 1}
    #algorithm = OwlSearch(dimension=dimension)
    # Optimizer initialization

    # HGSO
    hgso_params = {'n_cluster': 5, "cluster_size": 20, "alpha": 1.0, "beta": 1.0}
    algorithm = HGSO(dimension=dimension, **hgso_params)
    opt = Optimizer(obj.get_algorithm(), population, algorithm, hgso_callable)
    opt.run(100)
    #print(opt.population.global_optimum)
    plt.title(f"{obj.name}")
    opt.history.plot()
    plt.xlabel("Iteration")
    plt.ylabel("Optimal Value")
    plt.show()

