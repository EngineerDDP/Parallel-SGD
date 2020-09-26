from typing import List, Iterable
from nn import IOperator, ITrainable
from nn.layer import Dense
from nn.activation import Tanh, Softmax_NoGradient
from nn.model.abstract import Model


class DNN(Model):

    def __init__(self):
        super().__init__()
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        self.__var_list.extend(fc1.variables)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        self.__var_list.extend(fc2.variables)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)

        fc4 = Dense(inputs=fc3, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax_NoGradient(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5
