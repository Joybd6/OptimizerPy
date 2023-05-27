import numpy as np


class Population:
    _obj = None
    eval_value = None

    def __init__(self, lower_bound, upper_bound, size=10, dimension=10, objective_function=None):
        self.size = size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dimension = dimension
        self.population = None
        print(objective_function)
        self._obj = objective_function

    def initialize(self, initializer="uniform"):
        if initializer == "uniform":
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.size, self.dimension))
        elif initializer == "normal":
            self.population = np.random.normal(self.lower_bound, self.upper_bound, (self.size, self.dimension))
        else:
            raise ValueError("initializer must be either uniform or normal")

    @property
    def population(self):
        print("getter of population is called")
        return self._population

    @population.setter
    def population(self, value):
        if value is not None and value.shape != (self.size, self.dimension):
            raise ValueError("population shape must be (size, dimension)")

        self._population = value

        if self._population is None:
            self.eval_value = None

        if self._obj is not None and self._population is not None:
            print("objective function is not None")
            temp = np.empty(self.size)
            for i, agent in enumerate(self.population):
                temp[i] = self._obj(agent)
            self.eval_value = temp


if __name__ == "__main__":

    def sphere(x):
        return np.sum(x**2)

    pop = Population(-5, 5, 2, 2, objective_function=sphere)
    pop.initialize()

    print(pop.eval_value)
