import numpy as np

from time import sleep
from nn.interfaces import IOptimizer


class GradientDecentOptimizer_v2(IOptimizer):

    def __init__(self, learn_rate=0.01):

        self.LR = learn_rate
        self.Loss = None
        self.Layers = []

    def optimize(self, layers):
        self.Layers = layers

    def set_loss(self, loss):
        self.Loss = loss

    def forward_propagate(self, x):
        # forward propagation
        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))
        return intermediate

    def backward_propagate(self, intermediate, gradient):
        # backward propagation
        grad = gradient
        for i in range(1, len(self.Layers)+1):
            grad = self.Layers[-1*i].backpropagation(intermediate[-1 * (i+1)], grad)
        return None

    def calculate_gradient(self, y, label):
        return self.LR * self.Loss.gradient(y, label)

    def train(self, x, label):
        """
            train the network with labeled samples
        """
        # forward
        intermediate = self.forward_propagate(x)
        # apply learning rate
        grad = self.calculate_gradient(intermediate[-1], label)
        # backward
        self.backward_propagate(intermediate, grad)


class AdagradOptimizer(GradientDecentOptimizer_v2):

    def __init__(self, learn_rate=0.01):
        super().__init__(learn_rate)
        self.Gt = 0
        self.delta = 1e-8

    def train(self, x, label):
        """
            Adagrad training process.
        """
        # forward
        intermediate = self.forward_propagate(x)
        if self.Gt != 0:
            self.LR = self.LR / np.sqrt(self.Gt + self.delta)
        grad = self.calculate_gradient(intermediate[-1], label)
        # update Gt
        self.Gt = self.Gt + np.mean(np.square(grad))
        # backward
        self.backward_propagate(intermediate, grad)


class ParallelSGDOptimizer(GradientDecentOptimizer_v2):
    """
        Parallel Stochastic Gradient Descent Optimizer
    """

    def __init__(self, tags, com, batch_size, learn_rate=0.01):
        """
            Tags and communication thread helper are required for PSGD.
        :param tags: list of codec.tag.Tag
        :param com: instance of psgd.transfer.ITransfer
        :param learn_rate: scalar
        """
        super().__init__(learn_rate)
        self.Tags = tags
        self.TransferHelper = com
        self.Slice_To_Take = None
        self.BatchSize = batch_size

    def forward_propagate(self, x):
        """
            Forward propagate process.
        """
        # get multi-blocks
        blocks = []

        for tag in self.Tags:
            # save all blocks in one batch
            blocks.append(x[tag.getSliceWithinBatch()])

        if self.Slice_To_Take is None:
            self.Slice_To_Take = []
            start = 0
            end = 0
            for block in blocks:
                end = end + block.shape[0]
                self.Slice_To_Take.append(slice(start, end))
                start = end
        # take parts of x
        x = np.concatenate(blocks, axis=0)
        # continue calculation on partial data
        return super().forward_propagate(x)

    def calculate_gradient(self, y, label):
        """
            Calculated gradients on partial data.
        """
        block_labels = []

        for tag in self.Tags:
            # save all blocks in one batch
            block_labels.append(label[tag.getSliceWithinBatch()])

        label = np.concatenate(block_labels, axis=0)
        return super().calculate_gradient(y, label)

    def do_layer_wise_bp(self, x, layer, gradient):
        """
            do backward propagation layer-wise
        :param x: input of this layer
        :param layer: instance of ILayer
        :param gradient: gradient to the output of this layer
        """
        grad_back = []

        for j in range(len(self.Tags)):
            w, b, y = layer.delta_wb(x[self.Slice_To_Take[j]], gradient[self.Slice_To_Take[j]])
            grad_back.append(y)

            self.TransferHelper.put_weights(w, self.Tags[j], 'w')
            self.TransferHelper.put_weights(b, self.Tags[j], 'b')

        w_new = self.TransferHelper.get_weights(self.Tags[0], 'w') / self.BatchSize
        b_new = self.TransferHelper.get_weights(self.Tags[0], 'b') / self.BatchSize

        gradient = layer.apply_wb(w_new, b_new, np.concatenate(grad_back, axis=0))
        return gradient

    def backward_propagate(self, intermediate, gradient):
        """
            Backward propagation process.
        """
        for i in range(1, len(self.Layers)+1):
            nn = self.Layers[-1*i]
            gradient = self.do_layer_wise_bp(intermediate[-1*(i+1)], nn, gradient)
            # increase layer
            for tag in self.Tags:
                tag.incLayer()
        # increase batch
        for tag in self.Tags:
            tag.incBatch()

        return None


class DelayedPSGDOptimizer(ParallelSGDOptimizer):

    def __init__(self, tags, com, batch_size, learn_rate=0.01, delay_min=0, delay_max=2):
        super().__init__(tags, com, batch_size, learn_rate)
        self.Delay_Min = delay_min
        self.Delay_Max = delay_max

    def backward_propagate(self, intermediate, gradient):
        """
            Make some lags
        """
        sleep(np.random.uniform(self.Delay_Min, self.Delay_Max))
        return super().backward_propagate(intermediate, gradient)


class ParallelSGDWithPSOptimizer(ParallelSGDOptimizer):

    def __init__(self, tags, com, batch_size, learn_rate=0.01):
        super().__init__(tags, com, batch_size, learn_rate)
        # Save the initial value of each weights
        self.initial_value = None

    def forward_propagate(self, x):
        if self.initial_value is None:
            self.initial_value = {}
            for nn in self.Layers:
                self.initial_value[nn] = (nn.W.copy(), nn.B.copy())
        return super().forward_propagate(x)

    def do_layer_wise_bp(self, x, layer, gradient):
        """
            do backward propagation layer-wise
            using parameter server with initial value as zero.
            be aware, the initial value of parameter server is zero!!!
        :param x: input of this layer
        :param layer: instance of ILayer
        :param gradient: gradient to the output of this layer
        """
        for j in range(len(self.Tags)):
            block_x = x[self.Slice_To_Take[j]]
            block_grad = gradient[self.Slice_To_Take[j]]
            w, b, y = layer.delta_wb(block_x, block_grad)

            self.TransferHelper.put_weights(w / len(block_x), self.Tags[j], 'w')
            self.TransferHelper.put_weights(b / len(block_x), self.Tags[j], 'b')

        w_new = self.TransferHelper.get_weights(self.Tags[0], 'w')
        b_new = self.TransferHelper.get_weights(self.Tags[0], 'b')

        layer.W, layer.B = self.initial_value[layer][0] + w_new, self.initial_value[layer][1] + b_new

        gradient = layer.forward_gradient(x, gradient)
        return gradient