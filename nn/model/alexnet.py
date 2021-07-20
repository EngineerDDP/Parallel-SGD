from nn.activation import ReLU, Softmax
from nn.layer import Dense, Conv2D, MaxPool, Flatten
from nn.model.sequential import SequentialModel


def AlexNet():
    model = SequentialModel(input_shape=[-1, 227, 227, 3])
    model.add(Conv2D(strides=[4, 4], padding="VALID", kernel_size=[11, 11], kernel=96, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[5, 5], kernel=256, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=384, activation=ReLU()))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=384, activation=ReLU()))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=256, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Flatten())
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=1000, activation=ReLU()))
    model.add(Dense(units=10, activation=Softmax()))
    return model
