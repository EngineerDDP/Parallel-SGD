import os
import unittest

import numpy as np

import dataset
import dataset.transforms
import nn

os.chdir("../../")


class TestModel(unittest.TestCase):

    def test_cifarnet(self):
        model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
        model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5],
                                  activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5],
                                  activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5],
                                  activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5],
                                  activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
        model.add(nn.layer.Flatten())
        model.add(nn.layer.Dropout())
        model.add(nn.layer.Dense(units=128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model.compile(nn.gradient_descent.ADAMOptimizer(alpha=0.0001))

        model.save("CIFARNET.model")
        model = nn.model.Model.load("CIFARNET.model")
        os.remove("CIFARNET.model")

        trans = dataset.transforms.ImageCls()
        x, y, x_t, y_t = trans(*dataset.CIFAR().load())

        model.fit(x, label=y, epoch=1, batch_size=100)
        res = model.evaluate(x, y)
        self.assertGreater(res['accuracy'], 0.5)

    def test_dnn(self):
        model = nn.model.SequentialModel(input_shape=[-1, 784])
        model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
        model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model.compile(nn.gradient_descent.SGDOptimizer(0.05))

        data = dataset.MNIST()
        trans = dataset.transforms.Shuffle().add(dataset.transforms.ImageCls())

        x, y, x_t, y_t = trans(*data.load())

        model.fit(x, y, batch_size=64, epoch=2)
        res = model.evaluate(x_t, y_t)

        self.assertGreater(res['accuracy'], 0.93)

    def test_customized(self):
        class LR(nn.model.Model):

            def __init__(self):
                super().__init__()
                self.w = nn.Variable(shape=[1, 1])
                self.b = nn.Variable(shape=[1])

            def call(self, x):
                return x * self.w + self.b

            def trainable_variables(self):
                return self.w, self.b

        x = np.linspace(0, 5, 100).reshape([-1, 1])
        y = 2.718 * x + np.random.normal(scale=1, size=[100, 1])

        model = LR()

        model.setup(nn.loss.MSELoss(), nn.metric.MeanSquareError())
        model.compile(nn.gradient_descent.SGDOptimizer(learn_rate=0.1))

        model.fit(x, y, epoch=1000)

        res = model.evaluate(x, y)
        self.assertGreater(res['MSE'], 0.90)

    def test_alexnet(self):
        model = nn.model.SequentialModel(input_shape=[-1, 227, 227, 3])
        model.add(nn.layer.Conv2D(strides=[4, 4], padding="VALID", kernel_size=[11, 11],
                                  kernel=96, activation=nn.activation.ReLU()))
        model.add(nn.layer.MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
        model.add(nn.layer.Conv2D(strides=[1, 1], padding="SAME", kernel_size=[5, 5],
                                  kernel=256, activation=nn.activation.ReLU()))
        model.add(nn.layer.MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
        model.add(nn.layer.Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3],
                                  kernel=384, activation=nn.activation.ReLU()))
        model.add(nn.layer.Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3],
                                  kernel=384, activation=nn.activation.ReLU()))
        model.add(nn.layer.Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3],
                                  kernel=256, activation=nn.activation.ReLU()))
        model.add(nn.layer.MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
        model.add(nn.layer.Flatten())
        model.add(nn.layer.Dense(units=4096, activation=nn.activation.ReLU()))
        model.add(nn.layer.Dense(units=4096, activation=nn.activation.ReLU()))
        model.add(nn.layer.Dense(units=1000, activation=nn.activation.ReLU()))
        model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model.compile(nn.gradient_descent.ADAMOptimizer())
        model.save("AlexNet.test_model")

        self.assertTrue(os.path.exists("AlexNet.test_model"))
        os.remove("AlexNet.test_model")

    def test_lenet5(self):
        from dataset import MNIST
        from dataset.transforms import ImageCls, Shuffle

        # lenet-5
        model = nn.model.SequentialModel()
        model.add(nn.layer.Reshape(shape=[-1, 28, 28, 1]))
        model.add(nn.layer.Conv2D(kernel=6, kernel_size=[5, 5], strides=[1, 1], padding='SAME',
                                  activation=nn.activation.ReLU()))
        model.add(nn.layer.MaxPool(strides=[2, 2], size=[2, 2], padding='VALID'))

        model.add(nn.layer.Conv2D(kernel=16, kernel_size=[5, 5], strides=[1, 1], padding='VALID'
                                  , activation=nn.activation.ReLU()))
        model.add(nn.layer.MaxPool(strides=[2, 2], size=[2, 2], padding='VALID'))

        model.add(nn.layer.Conv2D(kernel=120, kernel_size=[5, 5], strides=[1, 1], padding='VALID'))

        model.add(nn.layer.Reshape(shape=[-1, 120]))
        model.add(nn.layer.Dense(84, activation=nn.activation.ReLU()))
        model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

        model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model.compile(optimizer=nn.gradient_descent.SGDOptimizer(learn_rate=0.01))

        data = MNIST()  # 使用MNIST数据集
        trans = Shuffle().add(ImageCls())  # 先对数据集做Shuffle操作，再对数据集进行像素分类处理

        train_x, train_y, test_x, test_y = trans(*data.load())
        model.fit(x=train_x, label=train_y, epoch=20, batch_size=64)
        # job = parallel.ParallelSGD(model, data, trans)
        # nodes = parallel.parse_worker(worker_cnt=1)
        #
        # job.parallel(nodes, codec=Plain, epoch=10)


if __name__ == '__main__':
    unittest.main()
