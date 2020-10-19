import nn
import numpy as np

import matplotlib.pyplot as plt


class LR(nn.model.Model):

    def __init__(self):
        super().__init__()
        self.w = nn.Variable(shape=[1,1])
        self.b = nn.Variable(shape=[1])

    def call(self, x):
        return x * self.w + self.b

    def trainable_variables(self):
        return self.w, self.b


if __name__ == '__main__':
    x = np.linspace(0, 5, 100).reshape([-1, 1])
    y = 2.718 * x + np.random.normal(scale=1, size=[100, 1])

    plt.plot(x, y, 'r.')
    # plt.show()

    model = LR()

    model.setup(nn.loss.MSELoss(), nn.metric.MeanSquareError())
    model.compile(nn.gradient_descent.SGDOptimizer(learn_rate=0.1))

    model.fit(x, y, epoch=1000)

    w, b = model.trainable_variables()

    print(w.get_value(), b.get_value())

    y_pre = w.get_value() * x + b.get_value()

    plt.plot(x, y_pre, 'b-')
    plt.show()
