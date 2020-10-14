from abc import ABCMeta, abstractmethod
from nn import ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from psgd.interface import ITransfer


class IOpContainer(metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, *variables: ITrainable):
        pass

    @abstractmethod
    def set_batch_size(self, batch_size: int):
        pass
