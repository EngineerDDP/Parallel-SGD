from neuralnetworks.layers import *
from neuralnetworks.model import *
from neuralnetworks.losses import *
from neuralnetworks.activations import *
from neuralnetworks.optimizer import *
from neuralnetworks.metrics import *
from log import Logger

from dataset.mnist_input import load_mnist

if __name__ == '__main__':

    model = SequentialModel_v2(logger=Logger('Test'))
    model.add(FCLayer_v2(128, act=Tanh()))
    model.add(FCLayer_v2(10, act=Sigmoid()))

    model.compile(optimizer=GradientDecentOptimizer_v2(),
                  loss=CrossEntropyLossWithSigmoid(),
                  metrics=[CategoricalAccuracy(), MeanSquareError(), RelativeError()])

    x, y = load_mnist(kind='train')
    model.fit(x, y, batch_size=64, epochs=10)

    x_test, y_test = load_mnist(kind='t10k')
    model.evaluate(x_test, y_test)

