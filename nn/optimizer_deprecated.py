from time import sleep

import numpy as np

from nn import IOptimizer
from nn.interface import IGDContainer, ILoss



class GradientDescentOptimizer(IGDContainer):
    """
        Packed Gradient Descent Optimizer.
        Use batch or no batch, 2 or 4 dimension inputs.
    """

    def __init__(self, learn_rate=0.01):
        self.__learn_rate = learn_rate
        self.__loss = None
        self.__ops = None

    def optimize(self, layers):
        self.__layers = [layer for layer in layers]
        self.__ops = [[(SGDOptimizer(var, self.__learn_rate)) for var in layer.Variables] for layer in layers]

    def set_loss(self, loss:ILoss):
        self.__loss = loss

    def train(self, x, label):
        # build placeholder
        layer_wise_output = [0 for _ in range(len(self.__layers)+1)]

        # forward propagate
        i = 0
        layer_wise_output[i] = x
        for layer in self.__layers:
            i += 1
            layer_wise_output[i] = layer(layer_wise_output[i-1])

        grad = self.__loss.gradient(layer_wise_output[-1], label)

        while i > 0:
            gx = self.__layers[i-1].gradient(layer_wise_output[i-1], grad)

            self.__ops[i].optimize(gx)


    def backward_propagate(self, intermediate, gradient):
        # backward_predict propagation
        grad = gradient
        for i in range(1, len(self.Layers) + 1):
            grad = self.Layers[-1 * i].backpropagation(intermediate[-1 * (i + 1)], grad)
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
        # backward_predict
        self.backward_propagate(grad)


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
        # backward_predict
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
            do backward_predict propagation layer-wise
        :param x: input_ref of this layer
        :param layer: instance of ILayer
        :param gradient: gradient to the output_ref of this layer
        """
        grad_back = []

        for j in range(len(self.Tags)):
            w, b, y = layer.gradient(x[self.Slice_To_Take[j]], gradient[self.Slice_To_Take[j]])
            grad_back.append(y)

            self.TransferHelper.put_weights(w, self.Tags[j], 'w')
            self.TransferHelper.put_weights(b, self.Tags[j], 'G')

        w_new = self.TransferHelper.get_weights(self.Tags[0], 'w') / self.BatchSize
        b_new = self.TransferHelper.get_weights(self.Tags[0], 'G') / self.BatchSize

        gradient = layer.apply_wb(w_new, b_new, np.concatenate(grad_back, axis=0))
        return gradient

    def backward_propagate(self, intermediate, gradient):
        """
            Backward propagation process.
        """
        for i in range(1, len(self.Layers) + 1):
            nn = self.Layers[-1 * i]
            gradient = self.do_layer_wise_bp(intermediate[-1 * (i + 1)], nn, gradient)
            # increase layer
            for tag in self.Tags:
                tag.incLayer()
        # increase batch
        for tag in self.Tags:
            tag.incBatch()

        return None


class FastParallelSGDOptimizer(ParallelSGDOptimizer):

    def __init__(self, tags, com, batch_size, learn_rate=0.01):
        super().__init__(tags, com, batch_size, learn_rate)

    def do_layer_wise_bp(self, x, layer, gradient):
        """
            Update weight and bias delta, but do training without update new weights.
        :param x: input_ref of this layer
        :param layer: instance of ILayer
        :param gradient: gradient to the output_ref of this layer
        """
        grad_back = []

        for j in range(len(self.Tags)):
            w, b, y = layer.gradient(x[self.Slice_To_Take[j]], gradient[self.Slice_To_Take[j]])
            grad_back.append(y)

            self.TransferHelper.put_weights(w, self.Tags[j], 'w')
            self.TransferHelper.put_weights(b, self.Tags[j], 'G')

        gradient = layer.apply_wb(0, 0, np.concatenate(grad_back, axis=0))
        return gradient

    def backward_propagate(self, intermediate, gradient):
        """
            Backward propagation process.
            Do weights update after the backward_predict propagate stage complete.
        """
        for i in range(1, len(self.Layers) + 1):
            nn = self.Layers[-1 * i]
            gradient = self.do_layer_wise_bp(intermediate[-1 * (i + 1)], nn, gradient)
            # increase layer
            for tag in self.Tags:
                tag.incLayer()

        # reset layer and do it again
        self.Tags[0].resetLayer()
        for i in range(1, len(self.Layers) + 1):
            nn = self.Layers[-1 * i]
            # get newest updates
            w_new = self.TransferHelper.get_weights(self.Tags[0], 'w') / self.BatchSize
            b_new = self.TransferHelper.get_weights(self.Tags[0], 'G') / self.BatchSize
            # apply data
            nn.apply_wb(w_new, b_new, np.asarray(0))
            # inc layer
            self.Tags[0].incLayer()
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


class ParaAverageOptimizer(ParallelSGDOptimizer):

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
            do backward_predict propagation layer-wise
            using parameter server with initial value as zero.
            be aware, the initial value of parameter server is zero!!!
        :param x: input_ref of this layer
        :param layer: instance of ILayer
        :param gradient: gradient to the output_ref of this layer
        """
        for j in range(len(self.Tags)):
            block_x = x[self.Slice_To_Take[j]]
            block_grad = gradient[self.Slice_To_Take[j]]
            w, b, y = layer.gradient(block_x, block_grad)

            self.TransferHelper.put_weights(w / len(block_x), self.Tags[j], 'w')
            self.TransferHelper.put_weights(b / len(block_x), self.Tags[j], 'G')

        w_new = self.TransferHelper.get_weights(self.Tags[0], 'w')
        b_new = self.TransferHelper.get_weights(self.Tags[0], 'G')

        layer.W, layer.B = self.initial_value[layer][0] + w_new, self.initial_value[layer][1] + b_new

        gradient = layer.forward_gradient(x, gradient)
        return gradient
