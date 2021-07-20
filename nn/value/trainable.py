from typing import Tuple

from numpy import ndarray

from nn import ITrainable, IOptimizer
from nn.value.abstract import AbsValue


class Weights(AbsValue, ITrainable):

    def __init__(self):
        super().__init__()
        self.__content = None
        self.__content_gradient = None
        self.__optimizer: [IOptimizer] = None

    def get_shape(self) -> Tuple[int]:
        return tuple(self.__content.shape)

    def get_value(self) -> ndarray:
        return self.__content

    def set_value(self, val) -> None:
        self.__content = val

    def get_gradient(self) -> ndarray:
        return self.__content_gradient

    def adjust(self, val):
        self.__content_gradient = val
        if self.__optimizer:
            self.__optimizer.optimize(self)

    def attach_optimizer(self, optimizer: IOptimizer) -> None:
        self.__optimizer = optimizer

    def reset(self) -> None:
        self.__content = None

    def __getstate__(self):
        return self.__content, self.id

    def __setstate__(self, state):
        self.__init__()
        self.var_id = state[1]
        self.__content = state[0]
