from abc import ABCMeta, abstractmethod

from numpy import ndarray

from rpc.communication import Communication
from log import IPrinter


class ITransfer(metaclass=ABCMeta):

    @abstractmethod
    def put_weights(self, content: ndarray, var_id: int, batch_no: int, block_id: int) -> None:
        """
            Put a intermediate weights.
            distributed to specified layer sgd processor (IParallelSGD).
        :param content: content of the weights : ndarray
        :param tag: description of the weights : codec.tag.Tag
        :param w_type: type of weights : str
        :return: None
        """
        pass

    @abstractmethod
    def get_weights(self, var_id: int, batch_no: int) -> ndarray:
        """
            Acquire intermediate weights from local layer sgd processor (IParallelSGD)
        :param tag: description of the weights : codec.tag.Tag
        :param w_type: type of the weights : str
        :return: content of the weights : ndarray
        """
        pass

    @abstractmethod
    def start_transfer(self, com: Communication, group_offset: int, printer: IPrinter, node_id: int) -> None:
        """
            Start transferring data between working process and
            network communication process.
        :param com: communication of rpc
        :param group_offset: id offset of this group, means the first id in this working group.
        :param printer: logger
        :param node_id: id of this worker
        :return:
        """
        pass
