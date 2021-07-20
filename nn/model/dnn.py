from typing import List, Iterable, Tuple
from nn import IOperator, ITrainable
from nn.layer import Dense, Dropout
from nn.activation import Tanh, Softmax
from nn.model.abstract import Model


class DNN(Model):

    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        self.__var_list.extend(fc1.variables)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        self.__var_list.extend(fc2.variables)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5
