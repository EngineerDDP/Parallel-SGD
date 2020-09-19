import numpy as np

from nn.abstract import AbsValue
from nn.interface import IVariable, IOptimizer, ITrainable
from nn.operation.abstract import OperandHelper


class Variable(OperandHelper, IVariable, ITrainable, AbsValue):
    """
        Variable for operation, trainable.
        Basic leaf node in computation graph.
    """
    def __init__(self, initial_value=None):
        super().__init__()
        self.__var = np.asarray(initial_value)
        self.__ref_gradient = None
        self.__attached_optimizer: IOptimizer = None

    # -------- Operator implementation --------

    @property
    def output_ref(self):
        return self.get_value()

    def output_shape(self) -> [list, tuple, None]:
        return self.get_shape()

    def forward_train(self):
        return self.__var

    def forward_predict(self):
        return self.__var

    def backward_train(self, arg) -> None:
        self.__ref_gradient = arg
        if self.__attached_optimizer:
            self.__attached_optimizer.optimize(self)

    def backward_predict(self, arg) -> None:
        self.__ref_gradient = arg
        return None

    # -------- Operator implementation --------

    # -------- Variable implementation --------

    def initialize(self, shape: list) -> None:
        high = np.sqrt(6 / (np.sum(shape)))
        low = -high
        self.__var = np.random.uniform(low=low, high=high, size=shape)
        return None

    # -------- Variable implementation --------

    # -------- Trainable implementation --------

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
        self.__var[:] = val
        return None

    def get_gradient(self):
        return self.__ref_gradient

    # -------- Value implementation --------


class Placeholder(OperandHelper, AbsValue):

    def __init__(self, shape=None):
        super().__init__()
        self.__hold = None
        self.__gradient_attachment = None
        self.set_shape(shape)

    def set_shape(self, shape):
        if isinstance(shape, list) or isinstance(shape, tuple):
            self.__input_shape = list(shape)
        elif isinstance(shape, int):
            self.__input_shape = [shape]
        elif shape is None:
            self.__input_shape = None
        else:
            raise AssertionError("Shape must be int, tuple or list.")
    # -------- Operator implementation --------

    @property
    def output_ref(self):
        return self.get_value()

    def output_shape(self) -> [list, tuple, None]:
        return self.get_shape() if self.get_shape() else None

    def forward_train(self):
        return self.__hold

    def forward_predict(self):
        return self.__hold

    def backward_train(self, arg) -> None:
        self.__gradient_attachment = arg

    def backward_predict(self, arg) -> None:
        self.__gradient_attachment = arg

    # -------- Operator implementation --------

    # -------- Value implementation --------

    def set_value(self, val=None) -> None:
        if val is not None:
            val = np.asarray(val)
            assert  self.__input_shape is None or list(val.shape[1:]) == self.__input_shape[1:], \
                "Given shape {} does't match with {}.".format(val.shape, self.output_shape())
            self.__hold = val
        else:
            self.__hold = np.random.normal(size=[2] + self.__input_shape)

    def get_value(self):
        return self.__hold

    def get_shape(self) -> list:
        return self.__input_shape

    def get_gradient(self):
        return self.__gradient_attachment

    # -------- Value implementation --------
