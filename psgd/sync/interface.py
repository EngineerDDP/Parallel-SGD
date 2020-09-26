from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Set

from numpy import ndarray


class OutdatedUpdates(Exception):

    def __init__(self):
        pass


class AsyncDetected(Exception):

    def __init__(self):
        pass


class ReadTimeOut(Exception):

    def __init__(self, retry_func):
        self.__retry = retry_func

    def retry(self):
        return self.__retry()


class IParallelSGD(metaclass=ABCMeta):
    """
        Working in one specified layer.
    """

    @abstractmethod
    def release_memory(self) -> None:
        """
            Release memory used by this sgd controller.
        :return:
        """
        pass

    @abstractmethod
    def update_weights(self, content: ndarray, batch_no: int, block_id: int) -> Iterable[dict]:
        """
            Update a calculated weights to this controller.
            Weights may be calculated throw partial samples.
        :return: json object to be sent throw network,
                 None if nothing needs to be sent.
        """
        pass

    @abstractmethod
    def require_weights(self, batch_no: int) -> ndarray:
        """
            Require a aggregated full calculated newest weights.
        :return: Weights Matrix: Numpy Array
        """
        pass

    @abstractmethod
    def accept_data(self, content: dict) -> [Iterable[dict]]:
        """
            Receive a decomposable object for local weights update.
            object received from local dispatcher
        :return: json object to be sent throw network,
                 None if nothing needs to be sent.
        """
        pass
