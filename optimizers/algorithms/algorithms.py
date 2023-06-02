from abc import ABC, abstractmethod


class Algorithms(ABC):
    _local_optimum_agent = None
    _local_worst_agent = None
    _global_optimum_agent = None
    _current_agent = None
    _max_iter = 1

    def __init__(self, name, dimension):
        self._name = name
        self.dimension = dimension

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if value < 0:
            raise ValueError("max iteration must be positive")
        if value is None:
            raise TypeError("max iteration cannot be None")
        self._max_iter = value

    @property
    def name(self):
        return self._name

    @abstractmethod
    def step(self, iteration):
        """This method is called in each population step"""
        pass

    @abstractmethod
    def update_algorithm_state(self, iteration):
        """This method is called in each population step"""

    @property
    def current_agent(self):
        return self._current_agent

    @current_agent.setter
    def current_agent(self, value):
        if value is None:
            raise ValueError("current agent cannot be None")
        if value.shape[0] != self.dimension:
            raise ValueError("current agent shape must be equal to (dimension,)")
        self._current_agent = value

    @property
    def local_optimum_agent(self):
        return self._local_optimum_agent

    @local_optimum_agent.setter
    def local_optimum_agent(self, value):
        if value is None:
            raise ValueError("local optimum agent cannot be None")
        if value.shape[0] != self.dimension:
            raise ValueError("local optimum agent shape must be equal to (dimension,)")
        self._local_optimum_agent = value

    @property
    def local_worst_agent(self):
        return self._local_worst_agent

    @local_worst_agent.setter
    def local_worst_agent(self, value):
        if value is None:
            raise ValueError("local worst agent cannot be None")
        if value.shape[0] != self.dimension:
            raise ValueError("local worst agent shape must be equal to (dimension,)")
        self._local_worst_agent = value

    @property
    def global_optimum_agent(self):
        return self._global_optimum_agent

    @global_optimum_agent.setter
    def global_optimum_agent(self, value):
        if value is None:
            raise ValueError("global optimum agent cannot be None")
        if value.shape[0] != self.dimension:
            raise ValueError("global optimum agent shape must be equal to (dimension,)")
        self._global_optimum_agent = value
