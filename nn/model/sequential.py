from typing import List, Tuple

from numpy import ndarray

from nn.model.interface import IModel
from nn import IOperator, IOptimizer, IMetric, ILoss, AbsLayer, ITrainable
from nn.data.interface import IDataFeeder
from nn.data.numpy_data_feeder import NumpyDataFeeder
from nn.optimizer import IOptimize
from utils.log import IPrinter
from nn.value.placeholder import Placeholder
from nn.model.abstract import FitResultHelper, Model


class SequentialModel(Model):

    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.__layers: List[AbsLayer] = []

    def add(self, layer:AbsLayer):
        self.__layers.append(layer)

    def pop(self):
        self.__layers.pop()

    def call(self, x:IOperator) -> IOperator:
        input = x
        for layer in self.__layers:
            layer.set_input(input)
            input = layer
        return input

    def trainable_variables(self) -> Tuple[ITrainable]:
        vars: List[ITrainable] = []
        for layer in self.__layers:
            vars.extend(layer.variables)
        return tuple(vars)

    def summary(self):

        summary = "\n------------\t\tModel Summary\t\t------------\n"
        for nn in self.__layers:
            nn:AbsLayer
            summary += '\t{}\t\t\n'.format(nn)
            summary += '\t\tInput:\t{};\n'.format([-1] + list(nn.input_ref.shape[1:]) if nn.input_ref is not None else "[Adjust]")
            summary += '\t\tOutput:\t{};\n'.format(nn.output_shape() if nn.output_shape() else "[Adjust]")

        if self.__loss:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.__loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.__optimizer)
            summary += '\tMetrics:\n'
            for metric in self.__metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'
        return summary
