import nn
from dataset import CIFAR
import pickle as p
import numpy as np

from dataset.mnist import MNIST
from dataset.transforms import ImageCls, Shuffle


if __name__ == '__main__':

    model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
    # model.add(nn.layer.Reshape(shape=[-1, 28, 28, 1]))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[3, 3], activation=nn.activation.LeakReLU()))
    model.add(nn.layer.Flatten())
    model.add(nn.layer.Dense(units=128, activation=nn.activation.HTanh()))
    model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

    model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

    model.compile(nn.optimizer.OpContainer(nn.optimizer.GDOptimizer, nn.gradient_descent.ADAMOptimizer))
    #
    model.save('abc.model')

    model = nn.model.Model.load('abc.model')
    model.compile(nn.optimizer.OpContainer(nn.optimizer.GDOptimizer, nn.gradient_descent.ADAMOptimizer, gd_params=(1e-4, )))

    trans = Shuffle().add(ImageCls())
    x, y, _, _ = trans(*CIFAR().load())
    model.fit(x, label=y, batch_size=100, epoch=10)
    model.save('abc.model')
    print('')

    # model = nn.model.SequentialModel(input_shape=[-1, 1])
    # model.add(nn.layer.Dense(units=1))
    # model.setup(nn.loss.MSELoss(), nn.metric.MeanSquareError())
    #
    # s1 = p.dumps(model)
    #
    # model.compile(nn.optimizer.OpContainer(nn.optimizer.GAOptimizer, nn.gradient_descent.ADAMOptimizer))
    #
    # s2 = p.dumps(model)
    #
    # model.fit(x=np.random.normal(size=[1000, 1]), label=np.random.normal(size=[1000, 1]), epoch=1, batch_size=1000)
    #
    # s3 = p.dumps(model)
    #
    # print('')