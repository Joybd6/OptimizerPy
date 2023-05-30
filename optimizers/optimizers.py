class Optimizer:
    """
    This class is the main class to hold all the building blocks of the optimization process.
    """

    def __init__(self, objective_function, population, algorithms, optimization="min", callable=None):
        """
        :param objective_function:
        :param population:
        :param algorithms:
        """
        self._obj = objective_function
        self._population = population
        self._algorithms = algorithms
        self._optimization = optimization

    def update_global(self):
        """
        This method updates the global optimum agent.
        :return:
        """
        pass

    def run(self):
        """
        This method runs the optimization process.
        :return:
        """
        pass
