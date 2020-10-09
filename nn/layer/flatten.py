import numpy as np

from typing import List, Tuple, Sequence
from nn.interface import IOperator
from nn.layer.abstract import Weights, AbsLayer
from nn.activation.interface import IActivation


class Flatten(AbsLayer):

    def __init__(self, activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__shape_in: [Sequence[int]] = None
        self.__shape_out: [Sequence[int]] = (-1, -1)

    @property
    def variables(self) -> tuple:
        return ()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        self.__shape_in = x.shape[1:]
        return np.reshape(x, (x.shape[0], -1))

    def do_forward_train(self, x):
        self.__shape_in = x.shape[1:]
        return np.reshape(x, (x.shape[0], -1))

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return np.reshape(grad, (-1, *self.__shape_in))

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_out

    def __str__(self):
        return "<Flatten Layer, shape: {}>".format(self.__shape_out)

    def __repr__(self):
        print(self.__str__())


if __name__ == '__main__':
    from nn.value import Variable
    x = Variable(shape=(2, 5, 5, 1))
    y = Flatten(inputs=x)
    print(y.F().shape)
    print(y.G(np.random.normal(size=(2,25))).shape)