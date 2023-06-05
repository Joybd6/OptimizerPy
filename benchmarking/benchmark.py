# Description: This file contains the benchmark functions for the optimization algorithms
# f1
import numpy as np

class Sphere():
    def __init__(self):
        self.name = "Sphere"
        self.dim = 30
        self.range = (-100,100)
        self.opt_value = 0.00
        self.optimization = 'minimize'

    def get_algorithm(self):
        def sphere_func(x):
            res = 0
            for i in x:
                res += i**2
            return res
        return sphere_func
# f2


class Schwefel_2_22():
    def __init__(self):
        self.name = "Schwefel_2_22"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-10, 10)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def schwefel_2_22_func(x):
            # Summation of xi
            #print(x)
            sum = 0
            for item in x:
                sum += item
            # Product of xi
            prod = 1
            for item in x:
                prod *= item
            res = sum + prod
            return res
        return schwefel_2_22_func
# f3


class Schwefel_1_2():
    def __init__(self):
        self.name = "Schwefel_1_2"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-100,100)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def schwefel_1_2_func(x):
            res = 0
            for item in range(len(x)):
                temp = 0
                for j in range(item+1):
                    temp += x[j]
                res += temp**2
            return res
        return schwefel_1_2_func
# f5

class Schwefel_2_26():
    def __init__(self):
        self.name = self.__class__.__name__
        self.dim = 30
        self.opt_value = -12569.5
        self.range = (-500, 500)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def schwefel_2_26_func(x):
            res = 0
            for item in x:
                res += -item * np.sin(np.sqrt(np.abs(item)))
        return schwefel_2_26_func


class Rosenbrock():
    def __init__(self):
        self.name = "Rosenbrock"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-30,30)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def rosenbrock_func(x):
            res = 0
            for i in range(len(x)-1):
                xi = x[i]
                xi_plus_1 = x[i + 1]
                xi_squared = xi ** 2
                term1 = 100 * (xi_plus_1 - xi_squared) ** 2
                term2 = (xi - 1) ** 2
                res += term1 + term2
            return res
        return rosenbrock_func

# f6


class Step():
    def __init__(self):
        self.name = "Step"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-100,100)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def step_func(x):
            res = 0
            for i in x:
                res += (i + 0.5) ** 2
            return res
        return step_func

# f7


class Quartic_with_noise():
    def __init__(self):
        self.name = "Quartic_with_noise"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-100, 100)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def quartic_with_noise_func(x):
            res = 0
            for i in range(len(x)):
                res += ((i+1)*(x[i]) ** 4) + np.random.rand()
            return res
        return quartic_with_noise_func

# f8
class Ackley():
    def __init__(self):
        self.name = "Ackley"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-32, 32)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def ackley_func(x):
            res = -20*np.exp(-0.2*np.sqrt(1/len(x)*sum(i**2 for i in x))) - \
                np.exp(1/len(x)*sum(np.cos(2*np.pi*i) for i in x))+20+np.exp(1)
            return res
        return ackley_func


class Rastrigin():
    def __init__(self):
        self.name = "Rastrigin"
        self.dim = 30
        self.opt_value = 0.00
        self.range = (-5.12, 5.12)
        self.optimization = 'minimize'

    def get_algorithm(self):
        def rastrigin_func(x):
            res = 10*len(x)+sum(i**2-10*np.cos(2*np.pi*i) for i in x)
            return res
        return rastrigin_func