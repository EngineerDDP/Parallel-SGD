import nn
from dataset import CIFAR
from dataset.transforms import ImageCls

if __name__ == '__main__':

    model = nn.model.SequentialModel(input_shape=[-1, 32, 32, 3])
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    model.add(nn.layer.Conv2D(kernel=64, kernel_size=[5, 5], activation=nn.activation.LeakReLU(leak_coefficient=0.2)))
    model.add(nn.layer.Flatten())
    model.add(nn.layer.Dense(units=128, activation=nn.activation.Tanh()))
    model.add(nn.layer.Dense(units=10, activation=nn.activation.Softmax()))

    trans = ImageCls()

    x, y, x_t, y_t = trans(*CIFAR().load())
    model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
    model.compile(nn.gradient_descent.ADAMOptimizer())
    model.fit(x, label=y, epoch=80, batch_size=100)
    model.evaluate(x, y)

