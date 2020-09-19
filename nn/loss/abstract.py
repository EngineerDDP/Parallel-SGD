from abc import abstractmethod

from nn.interface import IOperator


class AbsLoss(IOperator):

    def __init__(self, op1:IOperator, op2:IOperator):
        self.__op_left = op1
        self.__op_right = op2
        self.__ref_left = None
        self.__ref_right = None

    @abstractmethod
    def do_backward(self, arg1, arg2):
        """
            Do backward propagation.
        :return: tuple which contains gradients for both arg1 and arg2
        """
        pass

    @abstractmethod
    def do_forward(self, arg1, arg2):
        """
            Do forward propagation.
        :return: ndarray, Loss value.
        """
        pass

    def output_shape(self) -> [list, tuple, None]:
        return [-1, 1]

    def forward_train(self):
        self.__ref_left = self.__op_left.forward_train()
        self.__ref_right = self.__op_right.forward_train()
        return self.do_forward(self.__ref_left, self.__ref_right)

    def forward_predict(self):
        return self.do_forward(self.__op_left.forward_predict(), self.__op_right.forward_predict())

    def backward_train(self, arg=None) -> None:
        grad_left, grad_right = self.do_backward(self.__ref_left, self.__ref_right)
        self.__op_left.backward_train(grad_left)
        self.__op_right.backward_train(grad_right)

    def backward_predict(self, arg=None) -> None:
        self.backward_train(arg)
