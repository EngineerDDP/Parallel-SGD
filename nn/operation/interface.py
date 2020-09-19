from abc import ABCMeta, abstractmethod


class IActivation(metaclass=ABCMeta):

    @abstractmethod
    def do_forward(self, x):
        """
            Do forward predict.
        :param arg1: left argument
        :param arg2: right argument
        :return: result
        """
        pass

    @abstractmethod
    def do_backward(self, grad):
        """
            Do backward_predict propagate without adjusting variables.
            return gradients for both left and right arguments.
        :param grad: gradients from above operations.
        :return: tuple, contains gradients to left and right arguments.
        """
        pass
