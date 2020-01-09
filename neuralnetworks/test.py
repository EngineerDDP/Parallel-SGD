from neuralnetworks.layers import *
from neuralnetworks.model import *
from neuralnetworks.losses import *
from neuralnetworks.activations import *
from neuralnetworks.optimizer import *
from neuralnetworks.metrics import *
from log import Logger

from dataset.mnist_input import load_mnist

# if __name__ == '__main__':
#
#     model = SequentialModel_v2(logger=Logger('Test'))
#     model.add(FCLayer_v2(784, act=Tanh()))
#     model.add(FCLayer_v2(784, act=Tanh()))
#     model.add(FCLayer_v2(392, act=Tanh()))
#     model.add(FCLayer_v2(196, act=Tanh()))
#     model.add(FCLayer_v2(128, act=Tanh()))
#     model.add(FCLayer_v2(10, act=SoftmaxNoGrad()))
#
#     model.compile(optimizer=GradientDecentOptimizer_v2(learn_rate=0.05),
#                   loss=CrossEntropyLossWithSoftmax(),
#                   metrics=[CategoricalAccuracy(), MeanSquareError()])
#
#     model.summary()
#
#     x, y = load_mnist(kind='train')
#     model.fit(x, y, batch_size=128, epochs=10)
#
#     x_test, y_test = load_mnist(kind='t10k')
#     result = model.evaluate(x_test, y_test)
#
#     print('Evaluate result: {}'.format(dict(zip(model.History_Title[-len(result):], result))))


