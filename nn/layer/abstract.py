from typing import Tuple

import numpy as np

from abc import abstractmethod
from numpy import ndarray
from nn.activation.interface import IActivation
from nn.value.abstract import AbsValue
from nn.interface import IOperator, ITrainable, IOptimizer, ModelState
from nn.layer.interface import ILazyInitialization
from nn.activation.linear import Linear


class Weights(AbsValue, ITrainable):

    def __init__(self):
        super().__init__()
        self.__content = None
        self.__content_gradient = None
        self.__optimizer:[IOptimizer] = None

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

    def reset(self) -> None:
        high = np.sqrt(6 / np.sum(self.get_shape()))
        low = -high
        self.__content = np.random.uniform(low=low, high=high, size=self.get_shape())


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
        self.__activation = activation
        self.__initialized = False

    @property
    def input_ref(self):
        return self.__ref_input

    def set_input(self, input:IOperator):
        self.__op_input = input

    @property
    @abstractmethod
    def variables(self) -> Tuple[ITrainable]:
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
        self.__initialized = False

    def __forward_prepare(self, x):
        self.initialize_parameters(x)
        self.__initialized = True

    def F(self, x:[float, ndarray, tuple]=None, state:ModelState=ModelState.Training) -> [float, ndarray]:
        """
            Do forward propagate.
        :param x: input of this layer.
                This parameter works only when this layer is not part of the computation graph.
        :param state: State to identify training process, works in some particular layer like
                (Dropout).
        :return: output of this layer.
        """
        self.__ref_input = self.__op_input.F(x, state) if self.__op_input else x
        if not self.__initialized:
            self.__forward_prepare(self.__ref_input)
        if state != ModelState.Training:
            return self.__activation.do_forward(self.do_forward_predict(self.__ref_input))
        else:
            return self.__activation.do_forward(self.do_forward_train(self.__ref_input))

    def G(self, grad:[float, ndarray]=None) -> None:
        """
            Do backward and adjust parameters.
        :param grad: Gradients from back-propagation, set to None when this layer doesnt needs
                input gradients. e.g. loss functions.
        :return: None, try get gradients from placeholder or variable.
        """
        # adjust variables with given gradients.
        gradient = self.__activation.do_backward(None, grad)
        self.backward_adjust(gradient)
        # adjust previous layers.
        if self.__op_input:
            self.__op_input.G(self.backward_propagate(gradient))
