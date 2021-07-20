from nn.activation.interface import IActivation
from nn.abstract import AbsFlexibleUnaryNode


class AbsActivation(AbsFlexibleUnaryNode, IActivation):

    def output_shape(self):
        return None
