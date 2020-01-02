import numpy as np
import pandas as pd

import time
from log import Logger


class ModelFitWithMap:

    def __init__(self, nn, optimizer, onehot=True, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size):
        if batch_size > len(x):
            batch_size = len(x)

        batches = int(len(x) / batch_size)
        time_t = time.time()
        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)

                interval = time.time() - time_t

                if self.Debug and interval > -1:
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j + 1, epochs, i + 1, batches, loss))
                    # draw heat map of last layer, 2d only
                    w = self.NN[-1].W
                    samples = 300
                    w_1 = np.linspace(-0.1 + w[0,0], 0.1 + w[0,0], samples)
                    w_2 = np.linspace(-0.1 + w[0,1], 0.1 + w[0,1], samples)
                    loss_mech = np.zeros(shape=[samples, samples])

                    for a in range(samples):
                        for b in range(samples):
                            self.NN[-1].W = np.asarray([w_1[a], w_2[b]]).reshape(w.shape)
                            loss_mech[a][b] = self.Optimizer.loss(part_x, part_y)
                    self.NN[-1].W = w
                    grad, bias, _ = self.Optimizer.grad(part_x, part_y)
                    scale = 0.01
                    grad = w - self.Optimizer.LR * grad * scale
                    import matplotlib.pyplot as plt

                    w_1, w_2 = np.meshgrid(w_1, w_2)
                    fig = plt.figure()
                    plt.contourf(w_1, w_2, loss_mech, levels=7)
                    c = plt.contour(w_1, w_2, loss_mech, colors='black')
                    plt.clabel(c, inline=True, fontsize=10)
                    plt.plot([w[0,0], grad[0,0]], [w[0,1], grad[0,1]], 'r-')
                    plt.plot([grad[0,0]], [grad[0,1]], 'r>')
                    plt.savefig('./figs/epo{}-bat{}.png'.format(j+1, i+1))
                    plt.close(fig)

                    time_t = time.time()

        return loss

    def reset(self):

        for nn in self.NN:
            nn.reset()


    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x


class Trace_Model:

    def __init__(self, nn, optimizer, logger, onehot=True, trace_name='default'):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Log = logger
        self.Trace_File_Name = trace_name

    def fit(self, x, y, epochs, batch_size):

        track_content_time = []
        loss = 0

        if batch_size > len(x):
            batch_size = len(x)

        time_start = time.time()
        batches = int(len(x) / batch_size)

        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)
                val_acc = self.evalute(part_x, part_y)

                self.Log.log_message('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j + 1, epochs, i + 1, batches, loss))

                tracks = pd.DataFrame(data=track_content_time, index=None,
                                      columns=['time(s)', 'epoches', 'batches', 'total', 'loss', 'acc'])
                tracks.to_csv('./training/training_track_{}.csv'.format(self.Trace_File_Name))

                track_content_time.append([time.time() - time_start, j + 1, i + 1, j * batches + i, loss, val_acc])

        return loss

    def reset(self):

        for nn in self.NN:
            nn.reset()


    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x

    def evalute(self, x, y):

        predict = self.predict(x)
        loss = self.Optimizer.metric(x, y)
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        self.Log.log_message('Accuracy: {:.4f}, Total Loss: {:.4f}.'.format(acc, loss))

        return acc


class Normal_Model:

    def __init__(self, nn, optimizer, onehot=True, debug=True):
        self.NN = nn
        self.Optimizer = optimizer
        self.Onehot = onehot
        self.Debug = debug

    def fit(self, x, y, epochs, batch_size, val_x=None, val_y=None):

        loss = 0

        if batch_size > len(x):
            batch_size = len(x)

        time_t = time.time() - 6
        batches = int(len(x) / batch_size)

        # train
        for j in range(epochs):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                loss = self.Optimizer.train(part_x, part_y)

                interval = time.time() - time_t

                if self.Debug and interval > 5:
                    print('epochs: {}/{}, batches: {}/{}, loss: {:.4f}'.format(j + 1, epochs, i + 1, batches, loss))
                    self.evalute(part_x, part_y)

        return loss

    def reset(self):

        for nn in self.NN:
            nn.reset()


    def predict(self, x):

        # transpose x
        x = np.asmatrix(x).T

        for layer in self.NN:
            x = layer.F(x)

        x = x.T.getA()

        return x

    def evalute(self, x, y):

        predict = self.predict(x)
        loss = self.Optimizer.loss(x, y)
        if self.Onehot:
            y = y.argmax(axis=1)
            predict = predict.argmax(axis=1)
        else:
            predict = np.round(predict)

        acc = np.mean(np.equal(y, predict))

        if self.Debug:
            print('Accuracy: {:.4f}, Total Loss: {:.4f}.'.format(acc, loss))

        return acc


class SequentialModel_v2:

    def __init__(self, nn=None, logger=Logger('Default')):
        if nn is not None:
            self.NN = nn
        else:
            self.NN = []

        self.Optimizer = None
        self.Loss = None
        self.Log = logger
        self.Metrics = []
        self.History_Title = []
        self.History = []

    def add(self, unit):
        self.NN.append(unit)

    def pop(self):
        self.NN.pop()

    def compile(self, optimizer, loss, metrics):
        """
            Compile model.
        """
        # Get loss function
        self.Loss = loss
        # Add loss function to evaluate metrics
        self.Metrics.append(loss)
        # Get evaluate metrics
        self.Metrics.extend(metrics)
        self.History_Title = [metric.description() for metric in self.Metrics]
        # Set optimizer
        optimizer.optimize(self.NN)
        optimizer.set_loss(loss)
        self.Optimizer = optimizer

    def fit(self, x, y, batch_size, epochs):
        """
            Fit model parameters with given input samples.
        """
        if batch_size > len(x):
            batch_size = len(x)

        batches = len(x) // batch_size

        # train
        for j in range(epochs):
            for i in range(batches):

                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                self.Optimizer.train(part_x, part_y)
                eval_result = self.evaluate(part_x, part_y)
                str_output = ['{}:{:.4f}'.format(name, val) for name, val in zip(self.History_Title, eval_result)]
                self.Log.log_message('Epochs:{}/{}, Batches:{}/{}, Total batches:{}. {}'
                                     .format(j+1, epochs, i+1, batches, j*batches+i, ','.join(str_output)))

        return self.History

    def evaluate(self, x, y):

        predictions = self.predict(x)
        eval_results = [metric.metric(predictions, y) for metric in self.Metrics]
        self.History.append(eval_results)

        return eval_results

    def predict(self, x):

        intermediate = x

        for layer in self.NN:
            intermediate = layer.F(intermediate)

        return intermediate


