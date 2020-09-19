from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.loss.abstract import AbsLoss
from nn.metric.interface import IMetric
from nn.model.abstract import FitResultHelper
from nn.model.interface import IModel
from nn.optimizer import IOptimizer
from nn.variable import Placeholder
from utils.log import Logger


class SequentialModel(IModel):

    def __init__(self, input_shape=None):
        self.__placeholder_input = Placeholder(input_shape)
        self.__placeholder_output = Placeholder()
        self.__layers = []
        self.__variables = []
        self.__ref_output:[AbsLayer] = self.__placeholder_input
        self.__loss:[AbsLoss, None] = None
        self.__metrics = []
        self.__fit_history = FitResultHelper()
        self.__optimizer:[IOptimizer] = None

    def add(self, layer:AbsLayer):
        layer.set_input(self.__ref_output)
        self.__variables.extend(layer.variables)
        self.__layers.append(layer)
        self.__ref_output = layer

    def compile(self, optimizer:IOptimizer, loss:type, *metrics):
        # validate model
        if self.__placeholder_input.get_shape() is not None:
            self.__placeholder_input.set_value()
            # initialize and validate
            self.__ref_output.forward_predict()
        # set output_ref hooker
        self.__placeholder_output.set_shape(self.__ref_output.output_shape())
        # setup loss
        self.__loss = loss(self.__ref_output, self.__placeholder_output)
        # validate metrics and set title
        self.__metrics = metrics
        title = ["Epochs", "Batches", "in", "Total", "Loss"]
        for metric in self.__metrics:
            assert isinstance(metric, IMetric), "Something cannot be interpreted as metric were passed in."
            title.append(metric.description())

        # set title
        self.__fit_history.set_fit_title(title)
        # set optimizer
        optimizer.optimize(self.__variables)
        self.__optimizer = optimizer

    def evaluate_metrics(self, y, label) -> list:
        return [metric.metric(y, label) for metric in self.__metrics]

    def fit(self, x, label, batch_size, epoch, printer:Logger=None):
        assert isinstance(self.__loss, AbsLoss) and isinstance(self.__ref_output, IOperator), "Model hasn't complied."

        batch_size = min(batch_size, len(x))
        batches = len(x) // batch_size

        self.__optimizer.set_batch_size(batch_size)

        for j in range(epoch):
            for i in range(batches):
                start = i * batch_size % (len(x) - batch_size + 1)
                end = start + batch_size
                part_x = x[start:end]
                part_y = label[start:end]

                self.__placeholder_input.set_value(part_x)
                self.__placeholder_output.set_value(part_y)
                # do forward propagation to loss
                loss = self.__loss.forward_train()
                y = self.__ref_output.output_ref

                # do backward propagation from loss
                self.__loss.backward_train()
                # record fitting process
                fit_rec = [j+1, i+1, batches, self.__fit_history.count+1, loss]
                fit_rec.extend(self.evaluate_metrics(y, part_y))

                str_formatted = self.__fit_history.append_row(fit_rec)
                if printer:
                    printer.log_message(str_formatted)
                else:
                    print(str_formatted)

        return self.__fit_history

    def evaluate(self, x, label):
        # set placeholder
        self.__placeholder_input.set_value(x)
        self.__placeholder_output.set_value(label)
        # predict
        loss = self.__loss.forward_predict()
        y = self.__ref_output.output_ref
        # get evaluation
        eval = self.evaluate_metrics(y, label)
        eval.append(loss)
        # get title
        title = [metric.description() for metric in self.__metrics]
        title.append("Loss")
        return dict(zip(title, eval))

    def predict(self, x):
        self.__placeholder_input.set_value(x)
        y = self.__ref_output.forward_predict()
        return y

    def summary(self, printer:Logger=None):

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


if __name__ == '__main__':
    from nn.optimizer import Optimize, ADAMOptimizer, GDOptimizer, SGDOptimizer, GAOptimizer
    from nn.metric import MeanSquareError
    from nn.loss import MSELoss
    from nn.layer import Dense, Dropout
    from nn.operation import Sigmoid

    import numpy as np

    model = SequentialModel()
    fc1 = Dense(1, activation=Sigmoid())
    model.add(fc1)
    drop = Dropout()
    model.add(drop)
    fc2 = Dense(1)
    model.add(fc2)
    model.compile(Optimize(GDOptimizer, SGDOptimizer, op_params=(0.1,)), MSELoss, MeanSquareError())
    print(model.summary())
    model.fit(np.random.uniform(size=[10000, 10]), np.ones(shape=[10000, 1]), batch_size=64, epoch=20)
    print(model.summary())
    model.compile(Optimize(GAOptimizer, SGDOptimizer), MSELoss, MeanSquareError())
    model.fit(np.random.uniform(size=[10000, 10]), np.ones(shape=[10000, 1]), batch_size=640, epoch=1)