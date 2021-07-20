from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Set, Tuple

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


class ISyncType(metaclass=ABCMeta):
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
    def update_weights(self, content: ndarray, batch_no: int, block_id: int) -> Iterable[Tuple[List[int], dict]]:
        """
            Update a calculated weights to this controller.
            Weights may be calculated throw partial samples.
        :return: data for transmission
        """
        pass

    @abstractmethod
    def require_weights(self, batch_no: int) -> Tuple[ndarray, Iterable[Tuple[List[int], dict]]]:
        """
            Require a aggregated full calculated newest weights.
        :return: Weights Matrix: Numpy Array and other data for communication
        """
        pass

    @abstractmethod
    def accept_data(self, content: dict) -> Iterable[Tuple[List[int], dict]]:
        """
            Receive a decomposable object for local weights update.
            object received from local dispatcher
        :return: data for transmission
        """
        pass
