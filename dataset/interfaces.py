from abc import ABCMeta, abstractmethod
from enum import Enum, auto


class IDataset(metaclass=ABCMeta):

    @abstractmethod
    def load(self):
        """
            Load dataset.
        :return: train_x, train_y, test_x, test_y
        """
        pass

    @abstractmethod
    def check(self):
        """
            Check if dataset files were already available.
        :return: bool, indicates that there already has the dataset in local storage.
        """
        pass

class AbsDataset(IDataset, metaclass=ABCMeta):
    """
        IDataset interface, used for data transmission.
    """

    def __init__(self, check_sum=None):
        if check_sum == None:
            self.__md5_sum = self.check_sum()
        else:
            self.__md5_sum = check_sum

    def check(self) -> bool:
        """
            Check if dataset files were already available.
        :return: bool, indicates that there already has the dataset in local storage.
        """
        return self.check_sum() == self.__md5_sum

    @abstractmethod
    def load(self) -> tuple:
        """
            Load dataset.
        :return: train_x, train_y, test_x, test_y
        """
        pass

    @abstractmethod
    def check_sum(self) -> str:
        """
            Get check sum of dataset files.
        :return: bytes
        """
        pass

    @abstractmethod
    def extract_files(self) -> list:
        """
            Get associated filename list.
        :return: List of string
        """
        pass

    @abstractmethod
    def estimate_size(self) -> int:
        """
            Estimated dataset size.
        :return: int for bytes in size.
        """
        pass