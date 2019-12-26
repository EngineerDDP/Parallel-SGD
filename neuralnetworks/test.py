from neuralnetworks.layers import *
from neuralnetworks.model import *
from neuralnetworks.losses import *
from neuralnetworks.activations import *
from neuralnetworks.optimizer import *

from dataset import mnist_input

if __name__ == '__main__':

    x, y = mnist_input.load_mnist(kind='train')
    nn = [Reshape(shape=[28, 28, 1]),
          Conv2dLayer(filter_size=[5, 5], channel_count=6, padding='SAME', strikes=[1, 1], act=ReLU()),
          MaxPool(filter_size=[2, 2]),
          Conv2dLayer(filter_size=[5, 5], channel_count=16, padding='SAME', strikes=[1, 1], act=ReLU()),
          MaxPool(filter_size=[2, 2]),
          Reshape(shape=[7*7*16]),
          FCLayer_v2(120, act=Tanh()),
          FCLayer_v2(84, act=Tanh()),
          FCLayer_v2(10, act=Sigmoid())
          ]
    op = GradientDecentOptimizerv2(CrossEntropyLossWithSigmoid(), nn)
    model = Normal_Model(nn, op)
    model.fit(x, y, batch_size=64, epochs=2)

    model.evalute(x, y)
