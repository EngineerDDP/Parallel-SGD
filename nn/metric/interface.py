from abc import ABCMeta, abstractmethod


class IMetric(metaclass=ABCMeta):

    @abstractmethod
    def metric(self, y, label):
        """
            Calculate metrics value
        :return: Scala: single metrics value
        """
        pass

    @abstractmethod
    def description(self):
        """
            Official name for this metric.
        :return:
        """
        pass