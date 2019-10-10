import numpy as np

class Linear:
    """
    Linear activation
    """

    def __init__(self):
        pass

    def activation(self, x):
        return x

    def gradient(self, x):
        return 1


class ReLU:
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


class Sigmoid:
    """
    Sigmoid type activation
    """

    def __init__(self, delta=0.0):
        self.Delta = delta

    def activation(self, x):
        return 1 / (1 + np.exp(-1 * (x + self.Delta)))

    def gradient(self, x):
        return np.multiply(self.activation(x), (1 - self.activation(x)))


class Tanh:
    """
    Hyperbolic tangent function
    """

    def __init__(self, **kwargs):
        pass

    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.multiply(self.activation(x), self.activation(x))

