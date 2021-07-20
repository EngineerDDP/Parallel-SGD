# for general usage
from nn.operation.add import Add
from nn.operation.sub import Sub
from nn.operation.multiply import Multiply
from nn.operation.power import Power, Square
# tensorflow import guard
import importlib.util
_spec = importlib.util.find_spec("tensorflow")
if _spec is not None:
    from nn.operation.convolution import Conv2D
