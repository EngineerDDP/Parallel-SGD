import numpy as np
import matplotlib.pyplot as plt


class NoiseSimulation:

    def __init__(self, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.NScale = normal_scale
        self.BScale = bin_scale
        self.BRate = bin_rate
        self.Oneside = oneside

    def predict(self, x):
        # Gaussian noise
        n1 = np.random.normal(0.0, self.NScale, size=x.shape)
        # select points
        b1 = np.random.binomial(1, self.BRate, size=x.shape)
        if not self.Oneside:
            s1 = b1[np.where(b1 == 1)].shape
            # select side
            s1 = self.BScale * np.random.binomial(1, 0.5, size=s1)
            s1[np.where(s1 == 0)] = -1 * self.BScale
            # write back
            b1[np.where(b1 == 1)] = s1
        else:
            b1[np.where(b1 == 1)] = self.BScale

        return x + n1 + b1


class LinearSimulation:

    def __init__(self, w, b=0.0, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.W = w
        self.B = b

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):
        """
            Create samples with noise
        """

        return self.Noise.predict(np.dot(x, self.W.T) + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return np.dot(x, self.W) + self.B


class SinSimulation:

    def __init__(self, a=2.0, b=0.0, w=2*np.pi, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = sin(x * 2*pi/w) + b
        """

        self.B = b
        self.W = w
        self.A = a

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):

        return self.Noise.predict(np.sin(x * 2 * np.pi / self.W) + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return self.A * np.sin(x * 2 * np.pi / self.W) + self.B


def load(len_x:int=1024, len_y:int=1):

    x = np.random.uniform(0, 1, size=[60000, len_x])
    w = np.random.uniform(0, 1, size=[len_y, len_x])
    b = np.random.normal(0, 0.1, size=len_y)
    sim = LinearSimulation(w, b, normal_scale=0.3, bin_scale=1.0, bin_rate=0.1, oneside=False)

    y = sim.predict(x)

    return x[:50000], y[:50000], x[50000:], y[50000:]

def load_sin_sim():
    x_1 = np.linspace(-1, 1, 1000)
    x_2 = np.linspace(-1, 1.5, 1000)
    y = 0.2*np.sin(x_1) + 0.7*np.cos(x_2)
    y = y + np.random.normal(scale=0.1, size=x_1.shape)

    y = np.reshape(y, newshape=[-1, 1])
    x = np.asarray([x_1, x_2]).transpose()

    return x, y
