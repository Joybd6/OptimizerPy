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
from benchmarking.benchmark import *
# Import Optimizers
from optimizers import Optimizer

# Import Custom Optimizer
from optimizers.custom import HOptimizer
import matplotlib.pyplot as plt


#### Checking the algorithms

obj = Schwefel_1_2()

dimension = 30
pop_size = 100
upper_bound = np.array([obj.range[1]]).repeat(dimension)
lower_bound = np.array([obj.range[0]]).repeat(dimension)
algorithms = []
calls = []

# Sphere function



# Population initialization
population = Population(lower_bound, upper_bound, size=pop_size, dimension=dimension, objective_function=obj.get_algorithm(),
                        optimization="min")
population.initialize(initializer="uniform", lower_bound=lower_bound, upper_bound=upper_bound)

# Algorithms initialization
# HGSO
hgso_params = {'n_cluster': 5, "cluster_size": 20, "alpha": 1.0, "beta": 1.0}
hgso = HGSO(dimension=dimension, **hgso_params)
algorithms.append(hgso)
calls.append(hgso_callable)

# SineCos
sinecos_params = {'a': 0.8}
sinecos = SineCos(dimension=dimension, **sinecos_params)
algorithms.append(sinecos)
calls.append(sinecos_callable)

# Jaya
jaya_params = {}
jaya = Jaya(dimension=dimension, **jaya_params)
algorithms.append(jaya)
calls.append(jaya_callable)

# Butterfly
butterfly_params = {'c': 0.2, 'p': 0.8}
butterfly = Butterfly(dimension=dimension, **butterfly_params)
algorithms.append(butterfly)
calls.append(butterfly_callable)

# Owl Search
owl_search_params = {}
owl_search = OwlSearch(dimension=dimension, **owl_search_params)
algorithms.append(owl_search)
calls.append(owl_search_callable)

# HOptimizer
hoptimizer = HOptimizer(obj.get_algorithm(), population, algorithms, calls)
hoptimizer.run(100)
hoptimizer.history.name = f"Hybrid Henry Gas Solubility Optimization ({obj.name})"
hoptimizer.history.plot()
plt.show()
#print(hoptimizer.population.global_optimum)
#print("-----------------------------------")