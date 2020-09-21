from abc import ABCMeta, abstractmethod

from numpy import ndarray


class IActivation(metaclass=ABCMeta):

    @abstractmethod
    def do_forward(self, x: [float, ndarray], training: bool = True) -> [float, ndarray]:
        """
            Do forward propagation.
        """
        pass

    @abstractmethod
    def do_backward(self, x: [float, ndarray], grad: [float, ndarray]) -> [ndarray, float]:
        """
            Do backward propagation.
        """
        pass
