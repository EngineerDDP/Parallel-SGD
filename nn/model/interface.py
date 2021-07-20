from abc import ABCMeta, abstractmethod
from typing import Union, Type, Dict

from numpy import ndarray

from nn.data.interface import IDataFeeder
from nn.gradient_descent.interface import IGradientDescent
from nn.loss.abstract import ILoss
from nn.metric import IMetric
from nn.model.model_history import FitResultHelper
from nn.optimizer import IOpContainer
from log import IPrinter


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def setup(self, loss: ILoss, *metrics: IMetric):
        """
             loss and metrics
        :param loss: ILoss
        :param metrics: IMetric
        :return: None
        """
        pass

    @abstractmethod
    def compile(self, optimizer: Union[IOpContainer, Type[IGradientDescent]]):
        """
            Compile model with given optimizer
        :param optimizer: IOptimizer
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, x: [ndarray, IDataFeeder], label: [ndarray] = None, epoch: int = 4, batch_size: int = 64,
            printer: IPrinter = None) -> FitResultHelper:
        """
            Fit model with given samples.
        :param x: ndarray or data feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param epoch: int, Epoch of training
        :param label: ndarray, Label of samples
        :param batch_size: int, batch size
        :param printer: printer type
        :return: Fitting result, contains all history records.
        """
        pass

    @abstractmethod
    def fit_history(self) -> FitResultHelper:
        """
            Get all history records.
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self, x: ndarray, label: ndarray) -> Dict[str, float]:
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :return: evaluation result
        """
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """
            Predict give input
        :param x: input samples
        :return:
        """
        pass

    @abstractmethod
    def clear(self):
        """
            Clear and reset model parameters.
        :return:
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
            Return the summary string for this model.
        :return: String
        """
        pass
