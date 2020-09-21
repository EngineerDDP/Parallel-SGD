from abc import abstractmethod
from nn.interface import IOperator
from nn.operation.interface import IActivation


class OperandHelper(IOperator):

    # ---------------------------------------------------------------
    def __mul__(self, other: IOperator):
        """
            OP helper
        """
        from nn.operation.multiply import Multiply
        return Multiply(self, other)

    def __add__(self, other: IOperator):
        """
            OP helper
        """
        from nn.operation.add import Add
        return Add(self, other)

    def __sub__(self, other: IOperator):
        """
            OP helper
        """
        from nn.operation.sub import Sub
        return Sub(self, other)

    def __pow__(self, power, modulo=None):
        """
            OP helper
        """
        from nn.operation.power import Power
        return Power(self, power)
    # ---------------------------------------------------------------


class AbsBinaryOperator(OperandHelper):
    """
        Abs operation, basic module in the computation graph.
    """

    def __init__(self, op1, op2):
        # linked operation
        self.__left:IOperator = op1
        self.__right:IOperator = op2
        # reference for input_ref and output_ref
        self.__ref_left = None
        self.__ref_right = None
        self.__ref_output = None

    @property
    def op_left(self):
        return self.__left

    @property
    def op_right(self):
        return self.__right

    @property
    def left_input(self):
        return self.__ref_left

    @property
    def right_input(self):
        return self.__ref_right

    @property
    def output_ref(self):
        return self.__ref_output

    def forward_predict(self):
        """
            Do forward predict, without save reference for input_ref/output_ref parameters.
        :return: predictions
        """
        r_left = self.op_left.forward_predict()
        r_right = self.op_right.forward_predict()
        return self.do_forward(r_left, r_right)

    def forward_train(self):
        """
            Do forward propagate, saving input_ref and output_ref parameters for gradients calculation.
        :return: predictions
        """
        self.__ref_left = self.op_left.forward_train()
        self.__ref_right = self.op_right.forward_train()
        self.__ref_output = self.do_forward(self.__ref_left, self.__ref_right)
        return self.__ref_output

    def backward_train(self, arg) -> None:
        """
            Backward propagation.
        :param arg: parameters used in backward_predict propagation.
        :return: None
        """
        g_left, g_right = self.do_backward(arg)
        self.op_left.backward_train(g_left)
        self.op_right.backward_train(g_right)

    def backward_predict(self, arg) -> None:
        """
            Backward propagation.
        :param arg: parameters used in backward_predict propagation.
        :return: None
        """
        g_left, g_right = self.do_backward(arg)
        self.op_left.backward_predict(g_left)
        self.op_right.backward_predict(g_right)

    @abstractmethod
    def do_forward(self, arg1, arg2):
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


class AbsUnaryOperator(OperandHelper):

    def __init__(self, op:IOperator):
        self.__child_op = op
        self.__ref_input = None
        self.__ref_output = None

    @property
    def op_child(self):
        return self.__child_op

    @property
    def input_ref(self):
        return self.__ref_input

    @property
    def output_ref(self):
        return self.__ref_output

    def forward_train(self):
        self.__ref_input = self.__child_op.forward_train()
        self.__ref_output = self.do_forward(self.__ref_input)
        return self.__ref_output

    def forward_predict(self):
        return self.do_forward(self.__child_op.forward_predict())

    def backward_train(self, arg) -> None:
        self.__child_op.backward_train(self.do_backward(arg))

    def backward_predict(self, arg) -> None:
        self.__child_op.backward_predict(self.do_backward(arg))

    @abstractmethod
    def do_forward(self, x):
        """
            Do forward predict.
        :param arg: argument
        :return: result
        """
        pass

    @abstractmethod
    def do_backward(self, grad):
        """
            Do backward_predict propagate without adjusting variables.
            return gradients for both left and right arguments.
        :param grad: gradients from above operations.
        :return: gradient for continuing backward_predict propagation
        """
        pass
