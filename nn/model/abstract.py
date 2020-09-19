from abc import abstractmethod

from nn.interface import IOperator
from nn.loss.abstract import AbsLoss
from nn.variable import Placeholder
from nn.model.interface import IModel


class FitResultHelper:

    def __init__(self):
        self.__history_result = []
        self.__title = []

    @property
    def history(self):
        return self.__history_result

    @property
    def title(self):
        return self.__title

    @property
    def count(self):
        return len(self.__history_result)

    def set_fit_title(self, title):
        self.__title = title

    def append_row(self, row:list):
        eval_str = ', '.join(["{}:{:.4f}".format(key, val) if isinstance(val, float) else "{}:{}".format(key, val)
                              for key, val in zip(self.__title, row)])
        self.__history_result.append(row)
        return eval_str


class AbsModel(IModel):

    def __init__(self, input_shape=None, output_shape=None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__placeholder_output = Placeholder(output_shape)
        self.__ref_output:IOperator = None
        self.__ref_loss:AbsLoss = None

    @abstractmethod
    def trainable_variables(self) -> [list, tuple]:
        pass

    @abstractmethod
    def call(self, x:IOperator) -> IOperator:
        pass

    def compile(self, optimizer, loss, metrics):
        # get output_ref
        self.__ref_output = self.call(self.__placeholder_input)
        # do initialization if possible
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            self.__ref_output.forward_predict()
        # set output_ref hooker



    def fit(self, x, label, batch_size, epoch):
        pass

    def evaluate(self, x, label):
        pass

    def predict(self, x):
        pass

    def summary(self):
        pass

    def clear(self):
        pass
