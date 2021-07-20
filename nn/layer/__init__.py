# for general usage
from nn.layer.dense import Dense
from nn.layer.dropout import Dropout
from nn.layer.reshape import Reshape
from nn.layer.flatten import Flatten
from nn.layer.batchnorm import BatchNorm
# tensorflow import guard
import importlib.util
_spec = importlib.util.find_spec("tensorflow")
if _spec is not None:
    from nn.layer.conv2d import Conv2D
    from nn.layer.maxpool import MaxPool
