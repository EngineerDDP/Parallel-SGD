import numpy as np
import pandas as pd

import time
from log import Logger


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
        self.History_Title = ["Time", "Epoch", "Batch", "Total Batch"]
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
        self.History_Title.extend([metric.description() for metric in self.Metrics])
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
        time_started = time.time()

        # train
        for j in range(epochs):
            for i in range(batches):

                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = y[start:end]

                # do fitting
                self.Optimizer.train(part_x, part_y)
                eval_result = self.evaluate(part_x, part_y)

                # records time
                time_now = time.time()

                # log fitting progress
                str_output = ['{}:{:.4f}'.format(name, val) for name, val in zip(self.History_Title[-len(eval_result):], eval_result)]
                self.Log.log_message('Epochs:{}/{}, Batches:{}/{}, Total batches:{}. {}'
                                     .format(j+1, epochs, i+1, batches, j*batches+i, ','.join(str_output)))

                # record history data
                history = [time_now - time_started, j+1, i+1, j*batches+i+1]
                history.extend(eval_result)
                self.History.append(history)

        return self.History

    def evaluate(self, x, y):

        predictions = self.predict(x)
        eval_results = [metric.metric(predictions, y) for metric in self.Metrics]
        return eval_results

    def predict(self, x):

        intermediate = x

        for layer in self.NN:
            intermediate = layer.F(intermediate)

        return intermediate


