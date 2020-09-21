import numpy as np
from numpy import ndarray

from nn.value.abstract import AbsValue
from nn.interface import IOptimizer, ITrainable, ModelState
from nn.operation.abstract import OperandHelper


class Variable(AbsValue, OperandHelper, ITrainable):
    """
        Variable for operation, trainable.
        Basic leaf node in computation graph.
    """
    def __init__(self, initial_value=None, shape:[tuple, list]=None):
        super().__init__()
        self.__var = None
        if initial_value is not None:
            self.__var = np.asarray(initial_value)
        if shape:
            self.initialize(shape)
        self.__ref_gradient = None
        self.__attached_optimizer: [IOptimizer] = None

    def initialize(self, shape: [list, tuple]) -> None:
        high = np.sqrt(6 / (np.sum(shape)))
        low = -high
        self.__var = np.random.uniform(low=low, high=high, size=shape)

    # -------- Operator implementation --------

    @property
    def output_ref(self):
        return self.get_value()

    def output_shape(self) -> [list, tuple, None]:
        return self.get_shape()

    def F(self, x:[float, ndarray, tuple]=None, state:ModelState=ModelState.Training) -> [float, ndarray]:
        return self.__var

    def G(self, grad:[float, ndarray]=None) -> None:
        self.__ref_gradient = grad
        if self.__attached_optimizer:
            self.__attached_optimizer.optimize(self)

    # -------- Operator implementation --------

    # -------- Trainable implementation --------

    def reset(self) -> None:
        self.initialize(shape=self.__var.shape)

    def attach_optimizer(self, optimizer: IOptimizer) -> None:
        self.__attached_optimizer = optimizer
        return None

    # -------- Trainable implementation --------

    # -------- Value implementation --------

    def get_shape(self) -> list:
        return list(self.__var.shape) if self.__var is not None else None

    def get_value(self):
        return self.__var

    def set_value(self, val) -> None:
        self.__var = val
        return None

    def get_gradient(self):
        return self.__ref_gradient

    # -------- Value implementation --------

