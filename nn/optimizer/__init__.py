from typing import Type

from nn import ITrainable, IOptimizer
from nn.gradient_descent.interface import IGradientDescent
from nn.optimizer.interface import IOptimize


class Optimize(IOptimize):

    def __init__(self, optimizer_type: Type[IOptimizer], gradient_descent_type: Type[IGradientDescent], gd_params=tuple(), op_params=tuple()):
        self.__op_type = optimizer_type
        self.__op_params = op_params
        self.__gd_type = gradient_descent_type
        self.__gd_params = gd_params
        self.__op_list = []

    def optimize(self, *variables: ITrainable):
        for var in variables:
            self.__op_list.append(self.__op_type(self.__gd_type(*self.__gd_params), *self.__op_params))
            var.attach_optimizer(self.__op_list[-1])

    def set_batch_size(self, batch_size:int):
        for op in self.__op_list:
            op.set_batch_size(batch_size)

    def __str__(self):
        return "<Optimizer Container, (OP:{}), (GD:{})>".format(self.__op_type.__name__, self.__gd_type.__name__)


# for general usage
from nn.optimizer.gradient_descent import GDOptimizer
from nn.optimizer.gradient_ascent import GAOptimizer
from nn.optimizer.parallel_sgd import PSGDOptimizer
from nn.optimizer.double_buffering import DoubleBufferingOptimizer
from nn.optimizer.parameter_average import PAOptimizer
