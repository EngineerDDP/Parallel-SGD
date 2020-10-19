import time

from dataset import CIFAR, MNIST
from dataset.transforms import ImageCls, Shuffle
from nn.activation import Softmax, LeakReLU
from nn.gradient_descent import ADAMOptimizer
from nn.layer import Conv2D, MaxPool, Reshape, Dense, Flatten
from nn.loss import Cross_Entropy_With_Softmax
from nn.metric import CategoricalAccuracy
from nn.model import SequentialModel
from nn.optimizer import OpContainer, GDOptimizer

import nn
import numpy as np

if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # -------------------------------------------------------------------------
    # 逐层测试收敛，使其输入与输出均为1
    #
    # x = np.random.normal(loc=1, scale=0, size=(1, 32, 32, 1))
    #
    # inputs = nn.value.Placeholder()
    # inputs.set_value(x)
    # conv = nn.layer.Conv2DLayer(inputs=inputs, kernel=1, kernel_size=[5, 5], activation=LeakReLU())
    # conv = nn.layer.Conv2DLayer(inputs=conv, kernel=1, kernel_size=[5, 5], activation=LeakReLU())
    # Optimize(nn.optimizer.GDOptimizer, nn.gradient_descent.SGDOptimizer, gd_params=(1., )).optimize(*conv.variables)
    # while True:
    #     y_p = conv.F()
    #     conv.G(y_p - 1)
    #     print(np.square(y_p - 1).sum())
    #     # print(conv.variables[0].get_value().sum())
    #     time.sleep(0.5)

    # -------------------------------------------------------------------------
    # 模型测试

    model = SequentialModel()
    model.add(Reshape(shape=[-1, 28, 28, 1]))
    model.add(Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    model.add(Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    # # model.add(MaxPool(strides=[1, 2, 2, 1], padding="VALID", size=[2, 2]))
    model.add(Conv2D(kernel=64, kernel_size=[3, 3], activation=LeakReLU()))
    model.add(Conv2D(kernel=64, kernel_size=[3, 3], activation=LeakReLU()))
    # # model.add(MaxPool(strides=[1, 2, 2, 1], padding="VALID", size=[2, 2]))
    model.add(Flatten())
    # model.add(Dense(units=12800, activation=nn.activation.HTanh()))
    # model.add(Dense(units=12800, activation=nn.activation.HTanh()))
    # model.add(Dense(units=12800, activation=nn.activation.HTanh()))
    # model.add(Dense(units=12800, activation=nn.activation.HTanh()))
    model.add(Dense(units=128, activation=nn.activation.HTanh()))
    model.add(Dense(units=10, activation=Softmax()))

    trans = Shuffle().add(ImageCls())

    x, y, x_t, y_t = trans(*MNIST().load())
    x = x[:1000]
    y = y[:1000]
    x_t = x_t - 0.5
    model.setup(Cross_Entropy_With_Softmax(), CategoricalAccuracy())
    model.compile(nn.gradient_descent.ADAMOptimizer())
    for i in range(100):
        model.fit(x, label=y, epoch=10, batch_size=100)
        model.evaluate(x, y)

