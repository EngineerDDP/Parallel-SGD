from abc import ABCMeta, abstractmethod


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def compile(self, optimizer, loss, metrics):
        """
            Compile model with given optimizer, loss and metrics
        :param optimizer: IOptimizer
        :param loss: ILoss
        :param metrics: IMetric
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, x, label, batch_size, epoch):
        """
            Fit model with given samples.
        :param x:
        :param label:
        :return: fitting history
        """
        pass

    @abstractmethod
    def evaluate(self, x, label):
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :return: evaluation result
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
            Predict give input
        :param x: input samples
        :return:
        """
        pass

    @abstractmethod
    def summary(self):
        """
            Describe model structure.
        :return: None
        """
        pass

    @abstractmethod
    def clear(self):
        """
            Clear and reset model parameters.
        :return:
        """
        pass
