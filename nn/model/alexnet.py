from nn.activation import ReLU, Softmax_NoGradient
from nn.layer import Dense, Reshape, Conv2DLayer, MaxPool, Flatten
from nn.model.sequential import SequentialModel


def AlexNet():
    model = SequentialModel()
    model.add(Reshape(shape=[-1, 227, 227, 3]))
    model.add(Conv2DLayer(strides=[1, 4, 4, 1], padding="VALID", size=[11, 11, 3, 96], activation=ReLU()))
    model.add(MaxPool(strides=[1, 2, 2, 1], padding="VALID", size=[3, 3]))
    model.add(Conv2DLayer(strides=[1, 1, 1, 1], padding="SAME", size=[5, 5, 96, 256], activation=ReLU()))
    model.add(MaxPool(strides=[1, 2, 2, 1], padding="VALID", size=[3, 3]))
    model.add(Conv2DLayer(strides=[1, 1, 1, 1], padding="SAME", size=[3, 3, 256, 384], activation=ReLU()))
    model.add(Conv2DLayer(strides=[1, 1, 1, 1], padding="SAME", size=[3, 3, 384, 384], activation=ReLU()))
    model.add(Conv2DLayer(strides=[1, 1, 1, 1], padding="SAME", size=[3, 3, 384, 256], activation=ReLU()))
    model.add(MaxPool(strides=[1, 2, 2, 1], padding="VALID", size=[3, 3]))
    model.add(Flatten())
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=1000, activation=ReLU()))
    model.add(Dense(units=10, activation=Softmax_NoGradient()))
    return model
