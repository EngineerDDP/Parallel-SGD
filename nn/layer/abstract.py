from abc import abstractmethod

from nn.abstract import AbsValue
from nn.interface import IOperator, ITrainable, IOptimizer
from nn.layer.interface import ILazyInitialization
from nn.operation.interface import IActivation
from nn.operation.linear import Linear


class Weights(ITrainable, AbsValue):

    def __init__(self):
        super().__init__()
        self.__content = None
        self.__content_gradient = None
        self.__optimizer:IOptimizer = None

    def get_shape(self) -> list:
        return self.__content.shape

    def get_value(self):
        return self.__content

    def set_value(self, val) -> None:
        self.__content = val

    def get_gradient(self):
        return self.__content_gradient

    def adjust(self, val):
        self.__content_gradient = val
        if self.__optimizer:
            self.__optimizer.optimize(self)

    def attach_optimizer(self, optimizer:IOptimizer) -> None:
        self.__optimizer = optimizer


class AbsLayer(IOperator, ILazyInitialization):
    """
        Used for lazy initialization.
    """

    def __init__(self, input:IOperator=None, activation:IActivation=Linear()):
        """
            Abstract layer class
        :param input: input operator, IOperator instance
        """
        self.__op_input = input
        self.__ref_input = None
        self.__ref_output = None
        self.__activation = activation
        self.reset()

    @property
    def input_ref(self):
        return self.__ref_input

    @property
    def output_ref(self):
        return self.__ref_output

    def set_input(self, input:IOperator):
        self.__op_input = input

    @property
    @abstractmethod
    def variables(self) -> tuple:
        """
            Trainable units within this scope.
        :return: tuple
        """
        pass

    @abstractmethod
    def initialize_parameters(self, x) -> None:
        """
            Initialize parameters with given input_ref (x)
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_predict(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_train(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def backward_adjust(self, grad) -> None:
        """
            Backward propagate with weights adjusting.
        :param grad: ndarray
        """
        pass

    @abstractmethod
    def backward_propagate(self, grad):
        """
            Backward propagate.
        :param grad: ndarray
        :return: return the gradient from backward to input_ref (x)
        """
        pass

    def reset(self):
        self.__forward_predict_prepare = self.__forward_predict_prepare_prep
        self.__forward_train_prepare = self.__forward_train_prepare_prep

    def __forward_predict_prepare_prep(self, x):
        self.initialize_parameters(x)
        assert self.__op_input is not None, "This layer doesn't have input reference."
        # redirect function
        self.__forward_predict_prepare = self.do_forward_predict
        self.__forward_train_prepare = self.do_forward_train
        return self.__activation.do_forward(self.do_forward_predict(x))

    def __forward_train_prepare_prep(self, x):
        self.initialize_parameters(x)
        assert self.__op_input is not None, "This layer doesn't have input reference."
        # redirect function
        self.__forward_predict_prepare = self.do_forward_predict
        self.__forward_train_prepare = self.do_forward_train
        return self.__activation.do_forward(self.do_forward_train(x))

    def forward_predict(self):
        return self.__forward_predict_prepare(self.__op_input.forward_predict())

    def forward_train(self):
        self.__ref_input = self.__op_input.forward_train()
        self.__ref_output = self.__forward_train_prepare(self.__ref_input)
        return self.__ref_output

    def backward_predict(self, grad) -> None:
        grad = self.__activation.do_backward(grad)
        self.__op_input.backward_predict(self.backward_propagate(grad))
        self.backward_adjust(grad)

    def backward_train(self, grad) -> None:
        grad = self.__activation.do_backward(grad)
        self.backward_adjust(grad)
        self.__op_input.backward_predict(self.backward_propagate(grad))

