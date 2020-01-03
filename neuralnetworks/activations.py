import numpy as np

from abc import ABCMeta, abstractmethod


class IActivation(metaclass=ABCMeta):

    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass


class Linear(IActivation):
    """
    Linear activation
    """

    def __init__(self):
        pass

    def activation(self, x):
        return x

    def gradient(self, x):
        return 1


class ReLU(IActivation):
    """
    ReLU activation
    """
    def __init__(self):
        pass

    def activation(self, x):
        r = x.copy()
        r[r < 0] = 0
        return r

    def gradient(self, x):
        r = x.copy()
        r[r < 0] = 0
        r[r > 0] = 1
        return r


class Sigmoid(IActivation):
    """
    Sigmoid type activation
    """

    def __init__(self, delta=0.0):
        self.Delta = delta

    def activation(self, x):
        return 1 / (1 + np.exp(-1 * (x + self.Delta)))

    def gradient(self, x):
        return np.multiply(self.activation(x), (1 - self.activation(x)))


class SigmoidNoGrad(Sigmoid):
    """
        Sigmoid type activation, without gradient
    """
    def gradient(self, x):
        return 1


class Tanh(IActivation):
    """
    Hyperbolic tangent function
    """

    def __init__(self):
        pass

    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.multiply(self.activation(x), self.activation(x))


class SoftmaxNoGrad(IActivation):
    """
        Softmax activation, without gradient.
    """

    def __init__(self):
        pass

    def activation(self, x):
        denominator = np.sum(np.exp(x), axis=1).reshape([-1, 1])
        return np.exp(x) / denominator

    def gradient(self, x):
        return 1