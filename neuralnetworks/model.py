import time

from neuralnetworks.interfaces import ILayer, IOptimizer, IMetrics, ILoss, IActivation
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

    def add(self, unit:ILayer):
        self.NN.append(unit)

    def pop(self):
        self.NN.pop()

    def compile(self, optimizer:IOptimizer, loss:ILoss, metrics:[IMetrics]):
        """
            Compile model.
        """
        # Get loss function
        self.Loss = loss
        # Add loss function to evaluate metrics
        self.Metrics.append(loss)
        # Get evaluate metrics
        self.Metrics.extend(metrics)
        # Set title
        self.Evaluation_Title = [metric.description() for metric in self.Metrics]
        self.History_Title.extend(self.Evaluation_Title)
        # Set optimizer
        optimizer.optimize(self.NN)
        optimizer.set_loss(loss)
        self.Optimizer = optimizer

    def fit(self, x, y, batch_size:int, epochs:int):
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
                                     .format(j+1, epochs, i+1, batches, len(self.History)+1, ','.join(str_output)))

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

    def summary(self):

        self.Log.log_message('------------\t\tModel Summary\t\t------------\n')
        for nn in self.NN:
            self.Log.log_message('------------\t\t{}\t\t------------'.format(nn.__class__.__name__))
            self.Log.log_message('Input:\t{};'.format(nn.Input))
            self.Log.log_message('Output:\t{};'.format(nn.Output))

        if self.Loss is not None:
            self.Log.log_message('------------\t\tAppendix\t\t------------')
            self.Log.log_message('Loss:\t{}'.format(self.Loss.__class__.__name__))
            self.Log.log_message('Optimizer:\t{}'.format(self.Optimizer.__class__.__name__))
            self.Log.log_message('Metrics:\t')
            for metric in self.Metrics:
                self.Log.log_message('\t\t{}'.format(metric.__class__.__name__))
            self.Log.log_message('------------\t\tAppendix\t\t------------\n')

        self.Log.log_message('------------\t\tModel Summary\t\t------------')

    def clear(self):
        for nn in self.NN:
            nn.W = None
            nn.B = None



