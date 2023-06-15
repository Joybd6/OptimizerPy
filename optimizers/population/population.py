import numpy as np


class Population:
    _obj = None
    eval_value = None
    _local_optimum = None
    _global_optimum = None
    _callable_eval = None
    _population = None

    def __init__(self, lower_bound, upper_bound, size=10, dimension=10, objective_function=None, optimization='min'):
        self.size = size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dimension = dimension
        #self.population = None
        #print(objective_function)
        self._obj = objective_function
        self._optimization = optimization

    @property
    def local_optimum(self):
        return self._local_optimum

    @local_optimum.setter
    def local_optimum(self, value):
        raise ValueError("local optimum cannot be set through this method")

    @property
    def global_optimum(self):
        return self._global_optimum

    @global_optimum.setter
    def global_optimum(self, value):
        raise ValueError("global optimum cannot be set through this method")

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if value not in ['min', 'max']:
            raise ValueError("optimization must be either 'min' or 'max'")
        self._optimization = value

    def set_callable_on_eval(self, func):

        """
        This method sets the callable function to evaluate the population. Evaluate value will be passed to the function
        :param func:
        :return:
        """

        if isinstance(func, (list, tuple)):
            raise TypeError("func must be a list or tuple of function or callable objects")

        self._callable_eval = func

    def evaluate(self):
        """
        This method evaluates the population.
        :return:
        """
        if not callable(self._obj):
            raise ValueError("objective function must be callable")

        temp = np.empty(self.size)
        for i, agent in enumerate(self.population):
            temp[i] = self._obj(agent)

        self.eval_value = temp

        if self._callable_eval is not None:
            for i, call in enumerate(self._callable_eval):
                if not callable(call):
                    raise ValueError(f"callable_eval[{i}] must be callable")
                self.eval_value = call(self.eval_value)

    def update_local_optimum(self):
        """
        This method updates the local optimum agent.
        :return:
        """
        if self._optimization == 'min':
            self._local_optimum = self.population[np.argmin(self.eval_value)], np.min(self.eval_value)
        elif self._optimization == 'max':
            self._local_optimum = self.population[np.argmax(self.eval_value)], np.max(self.eval_value)
        else:
            raise ValueError("optimization must be either min or max")

    def update_global_optimum(self):
        """
        This method updates the global optimum agent.
        :return:
        """
        self.update_local_optimum()
        if self._optimization == 'min':
            if self._global_optimum is None or self._local_optimum[1] < self._global_optimum[1]:
                self._global_optimum = self._local_optimum
        elif self._optimization == 'max':
            if self._global_optimum is None or self._local_optimum[1] > self._global_optimum[1]:
                self._global_optimum = self._local_optimum

    def initialize(self, initializer="uniform", lower_bound=0, upper_bound=1):
        if initializer == "uniform":
            self.population = np.random.uniform(lower_bound, upper_bound, (self.size, self.dimension))
        elif initializer == "normal":
            self.population = np.random.normal(lower_bound, upper_bound, (self.size, self.dimension))
        else:
            raise ValueError("initializer must be either uniform or normal")

        self.evaluate()

    @property
    def population(self):
        # print("getter of population is called")
        return self._population

    @population.setter
    def population(self, value: np.ndarray):
        if value is None:
            raise TypeError("population cannot be None")

        if value is not None and value.shape != (self.size, self.dimension):
            raise ValueError("population shape must be (size, dimension)")

        self._population = np.clip(value, self.lower_bound, self.upper_bound)

        if self._obj is not None and self._population is not None:
            # print("objective function is not None")
            self.evaluate()


if __name__ == "__main__":
    def sphere(x):
        return np.sum(x ** 2)


    pop = Population(-5, 5, 2, 2, objective_function=sphere)
    pop.initialize()

    print(pop.eval_value)
