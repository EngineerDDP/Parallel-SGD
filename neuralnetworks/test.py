from neuralnetworks.layers import *
from neuralnetworks.model import *
from neuralnetworks.losses import *
from neuralnetworks.activations import *
from neuralnetworks.optimizer import *

from dataset import mnist_input

if __name__ == '__main__':

    x, y = mnist_input.load_mnist(kind='train')

