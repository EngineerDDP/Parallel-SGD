import nn
import dataset
import dataset.transforms


if __name__ == '__main__':

    model = nn.model.SequentialModel(input_shape=[-1, 784])
    model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
    model.add(nn.layer.Dense(128, activation=nn.activation.Tanh()))
    model.add(nn.layer.Dense(10, activation=nn.activation.Softmax()))

    model.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())

    # model.save("MNISTDNN.model")
    # # model = nn.model.Model.load('MNISTDNN.model')
    #
    data = dataset.MNIST()
    trans = dataset.transforms.Shuffle().add(dataset.transforms.ImageCls())

    x, y, x_t, y_t = trans(*data.load())

    model.compile(nn.gradient_descent.GradientDecay(0.1))
    model.fit(x, y, batch_size=64, epoch=10)
