from nn.activation.abstract import AbsActivation


class Linear(AbsActivation):

    def do_forward(self, x, training=True):
        return x

    def do_backward(self, x, grad):
        return grad
