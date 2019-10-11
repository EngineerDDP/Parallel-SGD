import numpy as np
from time import time

class GradientDecentOptimizer:

    def __init__(self, loss, layers, learnrate=0.01):

        self.LR = learnrate
        self.Loss = loss
        self.Layers = layers
        self.Grad = 0

    def loss(self, x, label):

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        return loss

    def train(self, x, label):
        """
            train the network with labeled samples
        """

        # reshape x to [-1,1]

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        # apply learning rate

        self.Grad = self.LR * self.Loss.gradient(intermediate[-1], label)
        grad = self.Grad

        # backward propagation

        self.Layers.reverse()
        i = 2
        for nn in self.Layers:
            grad = nn.backpropagation(intermediate[-1 * i], grad)
            i += 1

        self.Layers.reverse()

        # return loss

        return np.mean(loss)


class ParallelGradientDecentOptimizer(GradientDecentOptimizer):

    def __init__(self, loss, layers, tags, com, learnrate=0.01):
        """
            Distributed machine learning implementation
            tag: used to identify myself in cluster
            com: used to update and require parameters from cluster
        """
        GradientDecentOptimizer.__init__(self, loss, layers, learnrate)

        self.Tags = tags
        self.Com = com
        self.total_non_execution_time = 0

    def train(self, x, label):
        """
            train the network with labeled samples
        """

        total = len(x)

        # get multi-blocks
        blocks = []
        block_labels = []
        slices_to_take = []
        start = 0
        end = 0

        for tag in self.Tags:
            # save all blocks in one batch
            blocks.append(x[tag.getSliceWithinBatch()])
            block_labels.append(label[tag.getSliceWithinBatch()])

            # find block positions within batch
            end = end + blocks[0].shape[0]
            slices_to_take.append(slice(start, end))
            start = end

        x = np.concatenate(blocks, axis=0)
        label = np.concatenate(block_labels, axis=0)

        x = np.asmatrix(x).T
        label = np.asmatrix(label).T

        # forward propagation

        intermediate = [x]
        for nn in self.Layers:
            intermediate.append(nn.F(intermediate[-1]))

        loss = self.Loss.loss(intermediate[-1], label)

        # apply learning rate

        self.Grad = self.LR * self.Loss.gradient(intermediate[-1], label)
        grad = self.Grad

        # backward propagation

        self.Layers.reverse()
        i = 2
        for nn in self.Layers:

            for j in range(len(self.Tags)):
                # Note:列向量!
                w, b, y = nn.delta_wb(intermediate[-1 * i][:, slices_to_take[j]], grad[:, slices_to_take[j]])
                non_exec_start = time()
                self.Com.put_weights(w, self.Tags[j], 'w')
                self.Com.put_weights(b, self.Tags[j], 'b')
                non_exec_end = time()
                self.total_non_execution_time += non_exec_end - non_exec_start

            non_exec_start = time()

            w_new = self.Com.get_weights(self.Tags[0], 'w') / total
            b_new = self.Com.get_weights(self.Tags[0], 'b') / total

            non_exec_end = time()
            self.total_non_execution_time += non_exec_end - non_exec_start

            # w_new = w / total
            # b_new = b / total

            grad = nn.apply_wb(w_new, b_new, y)

            # increase layer
            for tag in self.Tags:
                tag.incLayer()
            i += 1

        self.Layers.reverse()

        # return loss
        for tag in self.Tags:
            tag.incBatch()

        # Release memory
        del x
        del label

        return np.mean(loss)


class AdagradOptimizer(GradientDecentOptimizer):

    def __init__(self, loss, layers, learnrate=0.01):
        super().__init__(loss, layers, learnrate)
        self.Gt = 0
        self.delta = 1e-8

    def train(self, x, label):
        # update learning rate
        learn_rate = self.LR
        if self.Gt != 0:
            self.LR = self.LR / np.sqrt(self.Gt + self.delta)

        # train
        loss = super().train(x, label)
        # print(self.LR)

        # update Gt
        self.Gt = self.Gt + np.mean(np.square(self.Grad))
        self.LR = learn_rate

        return loss

