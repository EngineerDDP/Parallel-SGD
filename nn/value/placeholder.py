import numpy as np

from numpy import ndarray
from nn.interface import ModelState
from nn.value.abstract import AbsValue
from nn.operation.abstract import OperandHelper


class Placeholder(OperandHelper, AbsValue):

    def __init__(self, shape=None):
        super().__init__()
        self.__hold = None
        self.__gradient_attachment = None
        self.__input_shape = None
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

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> [float, ndarray]:
        return self.__hold

    def G(self, grad: [float, ndarray] = None) -> None:
        self.__gradient_attachment = grad

    # -------- Operator implementation --------

    # -------- Value implementation --------

    def set_value(self, val=None) -> None:
        if val is not None:
            val = np.asarray(val)
            assert self.__input_shape is None or list(val.shape[1:]) == self.__input_shape[1:], \
                "Given shape {} does't match with {}.".format(val.shape, self.output_shape())
            self.__hold = val
        else:
            self.__hold = np.random.normal(size=[1] + self.__input_shape[1:])

    def get_value(self):
        return self.__hold

    def get_shape(self) -> list:
        return self.__input_shape

    def get_gradient(self):
        return self.__gradient_attachment

    # -------- Value implementation --------

    def __getstate__(self):
        return self.__input_shape

    def __setstate__(self, state):
        self.__init__(state)
