import sys
from abc import abstractmethod
from typing import Tuple, List, Iterable

import numpy as np
import pickle
from numpy import ndarray

from nn.data.interface import IDataFeeder
from nn.data.numpy_data_feeder import NumpyDataFeeder
from nn.interface import IOperator, IOptimizer, ITrainable, ModelState
from nn.loss.abstract import ILoss
from nn.metric import IMetric
from nn.model.interface import IModel
from nn.model.utils import FitResultHelper
from nn.optimizer import IOpContainer
from nn.value.placeholder import Placeholder
from utils.log import IPrinter


class Model(IModel):

    def __init__(self, input_shape: [Tuple[int]] = None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__ref_output: [IOperator] = None
        self.__metrics: List[IMetric] = []
        self.__loss: [ILoss] = None
        self.__optimizer: [IOptimizer] = None
        self.__fit_history: FitResultHelper = FitResultHelper()

    @abstractmethod
    def trainable_variables(self) -> Iterable[ITrainable]:
        pass

    @abstractmethod
    def call(self, x: IOperator) -> IOperator:
        pass

    @property
    def loss(self):
        return self.__loss

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def metrics(self):
        return self.__metrics

    def setup(self, loss: ILoss, *metrics: IMetric):
        if self.__ref_output is None:
            self.__ref_output = self.call(self.__placeholder_input)
        # validate model
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            # reset and validate
            self.__ref_output.F()
        # setup loss
        self.__loss: ILoss = loss
        # setup metric
        self.__metrics = [self.__loss]
        self.__metrics.extend(metrics)
        # validate metrics and set title
        title = ["Epochs", "Batches", "in", "Total"]
        for metric in self.__metrics:
            assert isinstance(metric, IMetric), "Something cannot be interpreted as metric were passed in."
            title.append(metric.description())

        # set title
        self.__fit_history.set_fit_title(title)

    def compile(self, optimizer: IOpContainer):
        # set optimizer
        self.__optimizer = optimizer
        self.__optimizer.optimize(*self.trainable_variables())

    def __evaluate_metrics(self, y, label) -> list:
        return [metric.metric(y, label) for metric in self.__metrics]

    def fit(self, x: [ndarray, IDataFeeder], epoch: int, label: [ndarray] = None, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        assert isinstance(self.__optimizer, IOpContainer), "Model hasn't complied."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."

        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(x.batch_size)
        title = [metric.description() for metric in self.__metrics]

        for j in range(epoch):
            epoch_rec = np.zeros(shape=[len(title)])
            for part_x, part_y in x:
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                batch_rec = self.__evaluate_metrics(y, part_y)
                fit_rec = [j + 1, x.position, x.length, self.__fit_history.count + 1]
                fit_rec.extend(batch_rec)
                epoch_rec += np.asarray(batch_rec) / x.length

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    # get stdout
                    sys.stdout.write('\r' + str_formatted)
                    sys.stdout.flush()
            print('')
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, epoch_rec)]
            print("Epoch Summary:{}".format(','.join(str_formatted)))

        return self.__fit_history

    def fit_history(self) -> FitResultHelper:
        return self.__fit_history

    def evaluate(self, x: ndarray, label: ndarray):
        assert isinstance(self.__loss, ILoss) and isinstance(self.__ref_output, IOperator), "Model hasn't setup."
        x = NumpyDataFeeder(x, label, batch_size=100)
        # get stdout
        import sys
        # get title
        title = [metric.description() for metric in self.__metrics]
        eval_recs = []
        for part_x, part_y in x:
            # set placeholder
            self.__placeholder_input.set_value(part_x)
            # do forward propagation
            y = self.__ref_output.F(state=ModelState.Evaluating)
            # get evaluation
            eval_rec = self.__evaluate_metrics(y, part_y)
            eval_recs.append(eval_rec)
            str_formatted = ["\t{}:{:.2f}".format(name, val) for name, val in zip(title, np.mean(eval_recs, axis=0))]
            sys.stdout.write("\rEvaluating: {:.2f}%{}.".format(100 * x.position / x.length, ','.join(str_formatted)))
            sys.stdout.flush()
        # flush a new line
        print('')
        return dict(zip(title, np.mean(eval_recs, axis=0)))

    def predict(self, x: ndarray):
        self.__placeholder_input.set_value(x)
        y = self.call(self.__placeholder_input).F(state=ModelState.Predicting)
        return y

    def clear(self):
        for var in self.trainable_variables():
            var.reset()

    def summary(self) -> str:

        summary = '\n------------\t\tModel Summary\t\t------------\n'

        summary += "No structure description available for this model.\n"

        if self.__loss and self.__optimizer and self.__metrics:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.__loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.__optimizer)
            summary += '\tMetrics:\n'
            for metric in self.__metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'

        return summary

    def save(self, file: str):
        with open(file, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(file: str) -> 'Model':
        with open(file, 'rb') as fd:
            model = pickle.load(fd)
        if model.__optimizer:
            model.compile(model.__optimizer)
        return model
