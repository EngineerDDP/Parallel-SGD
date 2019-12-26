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
        if self.Oneside == False:
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

        return self.Noise.predict(np.dot(x, self.W) + self.B)

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


def load_lin_sim():

    w = np.asarray([[0.2], [0.7]])
    b = 0.4
    x = np.random.normal(loc=1.0, scale=1.0, size=[500, 2])
    sim = LinearSimulation(w, b, 0.2)

    return x, sim.predict(x)

def load_sin_sim():
    x_1 = np.linspace(-1, 1, 1000)
    x_2 = np.linspace(-1, 1.5, 1000)
    y = 0.2*np.sin(x_1) + 0.7*np.cos(x_2)
    y = y + np.random.normal(scale=0.1, size=x_1.shape)

    y = np.reshape(y, newshape=[-1, 1])
    x = np.asarray([x_1, x_2]).transpose()

    return x, y

def show_fig_of_value_2d(func, sample_x, sample_y):

    x, y = np.meshgrid(sample_x, sample_y)
    val = np.zeros_like(x)

    for i in len(sample_x):
        for j in len(sample_y):
            val[j][i] = func(sample_x[i], sample_y[j])

    plt.contourf(x, y, val, levels=7)
    c = plt.contour(x, y, val, colors='black')
    plt.clabel(c, inline=True, fontsize=10)
    plt.show()


if __name__ == '__main__':
    w_1 = np.linspace(-1, 1, 100)
    w_2 = np.linspace(-1, 1, 100)

    x_1 = np.linspace(-1, 0.7, 100)
    x_2 = np.linspace(-0.6, 0.5, 100)
    y = 0.2*np.sin(x_1) + 0.7*np.cos(x_2)
    y = y + np.random.normal(scale=0.1, size=x_1.shape)


    def loss_sim(w_1, w_2):
        w = np.asarray([w_1, w_2])
        w = w.reshape([1, 2])
        x = np.asarray([x_1, x_2])
        # x = x.transpose()
        y_l = y.reshape([1, 100])
        ch = np.random.choice(100, 100)
        loss_r = 0
        for c in ch:
            loss_r += np.mean(np.square(np.tanh(np.dot(w, x[:, c]) - y_l[:, c])))
        return loss_r / len(ch)


    l = np.zeros([len(w_1),len(w_2)])
    for i in range(len(w_1)):
        for j in range(len(w_2)):
            l[i][j] = loss_sim(w_1[i], w_2[j])

    w_1_p, w_2_p = np.meshgrid(w_1, w_2)
    plt.contourf(w_1_p, w_2_p, l, levels=7)
    c = plt.contour(w_1_p, w_2_p, l, colors='black')
    plt.clabel(c, inline=True, fontsize=10)
    plt.show()

    from neuralnetworks.activations import Linear, Tanh
    from neuralnetworks.layers import FCLayer
    from neuralnetworks.model import Model, ModelFitWithMap
    from neuralnetworks.losses import MseLoss
    from neuralnetworks.optimizer import GradientDecentOptimizer

    nn = [
        FCLayer(units=2, act=Tanh()),
        FCLayer(units=1, act=Tanh())
    ]
    loss = MseLoss()
    op = GradientDecentOptimizer(loss, nn, 1)
    model = ModelFitWithMap(nn, op, False)
    x = np.asarray([x_1, x_2])
    model.fit(x.transpose(), y.reshape([-1, 1]), 1000, 100)

    pred = model.predict(x.transpose())
    pred = pred.reshape(-1)

    plt.plot(x_1, y, 'b-')
    plt.plot(x_1, pred, 'g.')
    plt.show()

    print(nn[0].W)
    print(nn[0].B)

