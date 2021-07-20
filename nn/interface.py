from abc import ABCMeta, abstractmethod
from enum import Enum, auto

from numpy import ndarray


class ModelState(Enum):
    Training = auto()
    Evaluating = auto()
    Predicting = auto()


class IOperator(metaclass=ABCMeta):
    """
        Operator interface.
        All operations and variables must implements this interface.
        This interface defined basic input and output functions for
        forward and backward propagation.
    """

    @abstractmethod
    def output_shape(self) -> [list, tuple, None]:
        """
            Output shape of this operator.
        """
        pass

    @abstractmethod
    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> [float, ndarray]:
        """
            Forward propagation.
        :param x: input source.
                Single ndarray input if this operator only accepts one argument.
                Input a tuple if this operator need multiple inputs.
        :param state: State of current process.
        :return: output of this layer.
        """
        pass

    @abstractmethod
    def G(self, grad: [float, ndarray]) -> None:
        """
            Backward propagate and update variables.
        :param grad: gradients of backward_predict layers
        """
        pass


class IBinaryNode(IOperator):

    @property
    @abstractmethod
    def op_left(self):
        pass

    @property
    @abstractmethod
    def op_right(self):
        pass


class IUnaryNode(IOperator):

    @property
    @abstractmethod
    def op_child(self):
        pass


class IFlexNode(metaclass=ABCMeta):
    """
        Flexible graph nodes interface.
        Make this operation can be operated with or without previous nodes.
    """

    @abstractmethod
    def set_input(self, *ops: IOperator):
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

    @abstractmethod
    def __getstate__(self):
        """
            Allow pickle this object.
        """
        pass

    @abstractmethod
    def __setstate__(self, state):
        """
            Allow pickle this object.
        """
        pass

    @property
    @abstractmethod
    def id(self):
        pass


class ITrainable(IValue):

    @abstractmethod
    def attach_optimizer(self, optimizer) -> None:
        """
            Attach a designated optimizer, replace the old one if possible.
        :param optimizer: new optimizer
        :return: None
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
            Initialize variable with given shape.
        :param shape: list
        :return: None
        """
        pass


class IOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, variables: ITrainable):
        """
            Register and optimize those parameters.
        :param variables: tuple contains var number of Variables.
        :return: None
        """
        pass

    @abstractmethod
    def set_batch_size(self, batch_size: int):
        """
            Set batch size for optimizer to optimize
        :param batch_size:
        :return:
        """
        pass
