from typing import List, Tuple

from nn import IOperator, AbsLayer, ITrainable
from nn.model.abstract import Model


class SequentialModel(Model):

    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.__layers: List[AbsLayer] = []

    def add(self, layer: AbsLayer):
        self.__layers.append(layer)

    def pop(self):
        self.__layers.pop()

    def call(self, x: IOperator) -> IOperator:
        inputs = x
        for layer in self.__layers:
            layer.set_input(inputs)
            inputs = layer
        return inputs

    def trainable_variables(self) -> Tuple[ITrainable]:
        var_list: List[ITrainable] = []
        for layer in self.__layers:
            var_list.extend(layer.variables)
        return tuple(var_list)

    def summary(self):

        summary = "\n------------\t\tModel Summary\t\t------------\n"
        for nn in self.__layers:
            nn: AbsLayer
            summary += '\t{}\t\t\n'.format(nn)
            summary += '\t\tInput:\t{};\n'.format(
                [-1] + list(nn.input_ref.shape[1:]) if nn.input_ref is not None else "[Adjust]")
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
