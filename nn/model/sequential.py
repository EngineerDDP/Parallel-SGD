from numpy import ndarray

from nn.model.interface import IModel
from nn import IOperator, IOptimizer, IMetric, ILoss, AbsLayer
from nn.data.interface import IDataFeeder
from nn.data.numpy_data_feeder import NumpyDataFeeder
from utils.log import IPrinter
from nn.value.placeholder import Placeholder
from nn.model.abstract import FitResultHelper


class SequentialModel(IModel):

    def __init__(self, input_shape=None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__layers = []
        self.__variables = []
        self.__ref_output:[AbsLayer] = self.__placeholder_input
        self.__loss:[ILoss, None] = None
        self.__metrics = []
        self.__fit_history = FitResultHelper()
        self.__optimizer:[IOptimizer] = None

    def add(self, layer:AbsLayer):
        layer.set_input(self.__ref_output)
        self.__variables.extend(layer.variables)
        self.__layers.append(layer)
        self.__ref_output = layer

    def compile(self, optimizer:IOptimizer, loss:ILoss, *metrics:IMetric):
        # validate model
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            # reset and validate
            self.__ref_output.F()
        # setup loss
        self.__loss:ILoss = loss
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
        # set optimizer
        self.__optimizer = optimizer
        self.__optimizer.optimize(self.__variables)

    def __evaluate_metrics(self, y, label) -> list:
        return [metric.metric(y, label) for metric in self.__metrics]

    def fit(self, x:[ndarray, IDataFeeder], epoch:int, label:[ndarray]=None, batch_size:int=64, printer:IPrinter=None) -> FitResultHelper:
        assert isinstance(self.__loss, ILoss) and isinstance(self.__ref_output, IOperator), "Model hasn't complied."
        assert isinstance(x, IDataFeeder) or label is not None, "Fitting process requires both x and label."

        if isinstance(x, ndarray):
            x = NumpyDataFeeder(x, label, batch_size=batch_size)

        self.__optimizer.set_batch_size(batch_size)

        for j in range(epoch):
            for part_x, part_y in x:
                self.__placeholder_input.set_value(part_x)
                # do forward propagation
                y = self.__ref_output.F()
                # get loss
                grad_y, _ = self.__loss.gradient(y, part_y)
                # do backward propagation from loss
                self.__ref_output.G(grad_y)
                # record fitting process
                fit_rec = [j+1, x.position, x.length, self.__fit_history.count+1]
                fit_rec.extend(self.__evaluate_metrics(y, part_y))

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    print(str_formatted)

        return self.__fit_history

    def evaluate(self, x, label):
        # set placeholder
        self.__placeholder_input.set_value(x)
        # do forward propagation
        y = self.__ref_output.F()
        # get lost
        loss = self.__loss.metric(y, label)
        # get evaluation
        eval = self.__evaluate_metrics(y, label)
        eval.append(loss)
        # get title
        title = [metric.description() for metric in self.__metrics]
        title.append("Loss")
        return dict(zip(title, eval))

    def predict(self, x):
        self.__placeholder_input.set_value(x)
        y = self.__ref_output.F()
        return y

    def summary(self):

        summary = "\n------------\t\tModel Summary\t\t------------\n"
        for nn in self.__layers:
            nn:AbsLayer
            summary += '\t{}\t\t\n'.format(nn)
            summary += '\t\tInput:\t{};\n'.format([-1] + list(nn.input_ref.shape[1:]) if nn.input_ref is not None else "[Adjust]")
            summary += '\t\tOutput:\t{};\n'.format(nn.output_shape() if nn.output_shape() else "[Adjust]")

        if self.__loss:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.__loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.__optimizer)
            summary += '\tMetrics:\n'
            for metric in self.__metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'
        return summary

    def clear(self):
        # reset to default, do initialization process again
        for nn in self.__layers:
            nn.reset()
