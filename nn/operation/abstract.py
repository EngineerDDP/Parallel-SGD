from nn.interface import IOperator


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
