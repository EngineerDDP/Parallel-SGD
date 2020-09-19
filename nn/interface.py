from abc import ABCMeta, abstractmethod


class IOperator(metaclass=ABCMeta):

    @abstractmethod
    def output_shape(self) -> [list, tuple, None]:
        """
            Output shape of this operator.
        """
        pass

    @abstractmethod
    def forward_train(self):
        """
            Forward propagate.
        :return: output_ref
        """
        pass

    @abstractmethod
    def forward_predict(self):
        """
            Forward propagate to get predictions.
        :return: output_ref
        """
        pass

    @abstractmethod
    def backward_train(self, arg) -> None:
        """
            Backward propagate and update variables.
        :param grad: gradients of backward_predict layers
        """
        pass

    @abstractmethod
    def backward_predict(self, arg) -> None:
        """
            Backward propagate without update variables.
        :param grad: gradients of backward_predict layers
        """
        pass


class IValue(metaclass=ABCMeta):

    @abstractmethod
    def get_shape(self) -> [list, None]:
        """
            Get shape of this value
        :return: list
        """
        pass

    @abstractmethod
    def get_gradient(self):
        """
            Get gradients from last calculation.
        :return: ndarray
        """
        pass

    @abstractmethod
    def get_value(self):
        """
            Get variable.
        :return: ndarray
        """
        pass

    @abstractmethod
    def set_value(self, val) -> None:
        """
            Set new value.
        :param val: new value, ndarray
        :return: None
        """
        pass

    @property
    @abstractmethod
    def id(self):
        pass


class IVariable(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self, shape:list) -> None:
        """
            Initialize variable with given shape.
        :param shape: list
        :return: None
        """
        pass


class IOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, *variables):
        """
            Register and optimize those parameters.
        :param variables: tuple contains var number of Variables.
        :return: None
        """
        pass

    @abstractmethod
    def set_batch_size(self, batch_size:int):
        """
            Set batch size for optimizer to optimize
        :param batch_size:
        :return:
        """
        pass


class ITrainable(metaclass=ABCMeta):

    @abstractmethod
    def attach_optimizer(self, optimizer:IOptimizer) -> None:
        """
            Attach a designated optimizer, replace the old one if possible.
        :param optimizer: new optimizer
        :return: None
        """
        pass
