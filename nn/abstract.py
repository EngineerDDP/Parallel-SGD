from abc import abstractmethod

from numpy import ndarray

from nn.interface import IOperator, IBinaryNode, IUnaryNode, ModelState, IFlexNode


class AbsFlexibleBinaryNode(IBinaryNode):

    def __init__(self, op1: [IOperator] = None, op2: [IOperator] = None):
        self.__op_left: IOperator = op1
        self.__op_right: IOperator = op2
        # input reference
        self.__ref_left: [float, ndarray] = 0
        self.__ref_right: [float, ndarray] = 0

    @property
    def op_left(self):
        return self.__op_left

    @property
    def op_right(self):
        return self.__op_right

    def set_child(self, op1: IOperator, op2: IOperator):
        self.__op_left = op1
        self.__op_right = op2

    @abstractmethod
    def do_forward(self, left: [float, ndarray], right: [float, ndarray], training: bool = True) -> [float, ndarray]:
        """
            Do forward propagation.
        """
        pass

    @abstractmethod
    def do_backward(self, left: [float, ndarray], right: [float, ndarray], grad: [float, ndarray]) -> (
            ndarray, ndarray):
        """
            Do backward propagation.
        """
        pass

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> [float, ndarray]:
        """
            Forward propagate to get predictions.
        :return: output_ref
        """
        assert isinstance(x, tuple) or x is None, "This operator requires tuple inputs, but ({}) was given".format(
            type(x))
        if self.op_left:
            self.__ref_left = self.op_left.F(x[0] if x else None, state)
        else:
            self.__ref_left = x[0]
        if self.op_right:
            self.__ref_right = self.op_right.F(x[1] if x else None, state)
        else:
            self.__ref_right = x[1]
        return self.do_forward(self.__ref_left, self.__ref_right, state == ModelState.Training)

    def G(self, grad: [float, ndarray] = None) -> None:
        """
            Backward propagate and update variables.
        :param grad: gradients of backward_predict layers
        """
        grad_left, grad_right = self.do_backward(self.__ref_left, self.__ref_right, grad)
        if self.op_left:
            self.op_left.G(grad_left)
        if self.op_right:
            self.op_right.G(grad_right)

    def clear_unused(self):
        pass

    def __getstate__(self):
        self.__ref_right = 0
        self.__ref_left = 0
        self.clear_unused()
        return self.__dict__


class AbsFlexibleUnaryNode(IUnaryNode, IFlexNode):

    def __init__(self, op: [IOperator] = None):
        self.__op_child: IOperator = op
        self.__ref_input: [ndarray, float] = 0

    @property
    def op_child(self):
        return self.__op_child

    def set_input(self, op: IOperator):
        self.__op_child = op

    @abstractmethod
    def do_forward(self, x: [float, ndarray], training: bool = True) -> [float, ndarray]:
        """
            Do forward propagation.
        """
        pass

    @abstractmethod
    def do_backward(self, x: [float, ndarray], grad: [float, ndarray]) -> [ndarray, float]:
        """
            Do backward propagation.
        """
        pass

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> [float, ndarray]:
        """
            Forward propagate to get predictions.
        :return: output_ref
        """
        if self.__op_child:
            self.__ref_input = self.op_child.F(x, state)
        else:
            self.__ref_input = x
        self.do_forward(self.__ref_input, state == ModelState.Training)

    def G(self, grad: [float, ndarray] = None) -> None:
        """
            Backward propagate and update variables.
        :param grad: gradients of backward_predict layers
        """
        grad_back = self.do_backward(self.__ref_input, grad)
        if self.__op_child:
            self.op_child.G(grad_back)

    def clear_unused(self):
        pass

    def __getstate__(self):
        self.__ref_input = 0
        self.clear_unused()
        return self.__dict__
