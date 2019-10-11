import numpy as np

from neuralnetworks.activations import Linear

class FCLayer:

    def __init__(self, units, w_init=None, b_init=None, act=Linear()):

        # use lazy initialization
        if w_init is not None:
            self.W = w_init
        else:
            self.W = None

        if b_init is not None:
            self.B = b_init
        else:
            self.B = None

        self.Act = act
        self.Output = units

    def reset(self):
        self.W = None
        self.B = None

    def logit(self, x):
        """
            Calculate logit
        """
        # lazy initialization
        # w shape is [output, input]
        # b shape is [output]
        if self.W is None:
            high = np.sqrt(1 / x.shape[0])
            low = -high
            self.W = np.random.uniform(low=low, high=high, size=[self.Output, x.shape[0]])
        if self.B is None:
            self.B = np.zeros(shape=[self.Output, 1])

        return np.dot(self.W, x) + self.B

    def F(self, x):
        """
            output function
        """
        # activation
        return self.Act.activation(self.logit(x))

    def delta_wb(self, x, gradient):
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y = np.multiply(act_grad, gradient)

        # adjust weight
        delta_weight = y.dot(x.T)
        delta_bias = y.sum(axis=1)

        return delta_weight, delta_bias, y

    def apply_wb(self, w, b, y):

        self.W = self.W - w
        self.B = self.B - b

        # calculate gradient for BP
        grad = self.W.transpose().dot(y)

        return grad

    def backpropagation(self, x, gradient):
        """
            Calculate gradient, adjust weight and bias and return gradients of this layer
            x shape=[input, samples count]
            grad shape=[output, samples count]
        """
        # calculate gradient
        act_grad = self.Act.gradient(self.logit(x))
        # y shape=[output, samples count]
        y = np.multiply(act_grad, gradient)

        # adjust weight
        batch_weight = y.dot(x.T) / y.shape[1]
        self.W = self.W - batch_weight
        # adjust bias
        self.B = self.B - y.mean(axis=1)
        # recalculate gradient to propagate
        grad = self.W.transpose().dot(y)
        return grad
