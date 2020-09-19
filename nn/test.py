from nn.variable import Placeholder
from nn.variable import Variable
import numpy as np
w = Variable(np.ones(shape=[2,2]))
w.get_value()
x = Placeholder(shape=[2])
b = Variable(np.zeros(shape=[2]))
b.get_value()
y = x * w + b
x.set_value(np.ones(shape=[10, 2]))
