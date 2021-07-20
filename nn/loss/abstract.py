from abc import abstractmethod

from numpy import ndarray

from nn.metric import IMetric


class ILoss(IMetric):

    @abstractmethod
    def gradient(self, left: [float, ndarray], right: [float, ndarray]) -> (ndarray, ndarray):
        """
            Calculated gradient of L(x, y) for both x and y.
        :param left: left input
        :param right: right input
        :return: tuple for left gradient and right gradient
        """
        pass

    def description(self):
        return "Loss"
