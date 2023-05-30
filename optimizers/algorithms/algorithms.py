from abc import ABC, abstractmethod


class Algorithms(ABC):
    _local_optimum_agent = None
    _local_worst_agent = None
    _global_optimum_agent = None
    _current_agent = None

    def __init__(self, name, dimension, max_iter):
        self._name = name
        self.max_iter = max_iter
        self.dimension = dimension

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

    def set_agents(self, current_agent, local_optimum_agent=None, local_worst_agent=None,
                   global_optimum_agent=None):
        if current_agent.shape[0] != self.dimension:
            raise ValueError("current agent shape must be equal to (dimension,)")

        if local_optimum_agent is not None and local_optimum_agent.shape[0] != self.dimension:
            raise ValueError("local optimum agent shape must be equal to (dimension,)")

        if local_worst_agent is not None and local_worst_agent.shape[0] != self.dimension:
            raise ValueError("local worst agent shape must be equal to (dimension,)")

        if global_optimum_agent is not None and global_optimum_agent.shape[0] != self.dimension:
            raise ValueError("global optimum agent shape must be equal to (dimension,)")

        self._current_agent = current_agent
        self._local_optimum_agent = local_optimum_agent
        self._local_worst_agent = local_worst_agent
        self._global_optimum_agent = global_optimum_agent
