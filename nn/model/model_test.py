import os
import unittest

import nn
from dataset import CIFAR
from dataset.transforms import ImageCls


class TestSequentialModel(unittest.TestCase):

    def test_cifarnet(self):
        os.chdir("../../")

        model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
        model.add(
            nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(
            nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(
            nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(
            nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(nn.layer.Flatten())
        model.add(nn.layer.Dropout())
        model.add(nn.layer.Dense(units=128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model.compile(nn.gradient_descent.ADAMOptimizer(alpha=0.0001))

        model.save("CIFARNET.model")

        trans = ImageCls()
        x, y, x_t, y_t = trans(*CIFAR().load())

        model.fit(x, label=y, epoch=1, batch_size=100)
        res = model.evaluate(x, y)
        print(res)
        # self.assertGreater(res['accuracy'], 0.5)

        os.remove("CIFARNET.model")
