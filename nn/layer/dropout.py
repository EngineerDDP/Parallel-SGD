import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.abstract import IActivation


class Dropout(AbsLayer):

    def __init__(self, drop_out_rate: float = 0.5, activation:IActivation=None, input=None):
        super().__init__(input, activation)
        self.__ref_mask = None
        self.__probability = drop_out_rate

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        return x

    def do_forward_train(self, x):
        self.__ref_mask = np.random.uniform(0, 1, size=x.shape) > self.__probability
        return np.multiply(x, self.__ref_mask)

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return np.multiply(grad, self.__ref_mask)

    def output_shape(self) -> [list, tuple, None]:
        return None

    def __str__(self):
        return "<Dropout Layer, Drop probability: {}>".format(self.__probability)

    def __repr__(self):
        print(self.__str__())
