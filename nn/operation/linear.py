from nn.operation.interface import IActivation


class Linear(IActivation):

    def do_forward(self, x):
        return x

    def do_backward(self, grad):
        return grad