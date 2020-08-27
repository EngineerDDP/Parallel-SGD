from abc import ABCMeta, abstractmethod


class IServerModel(metaclass=ABCMeta):

    @property
    @abstractmethod
    def weights_types(self):
        pass

    @abstractmethod
    def get_nn(self):
        pass

    @property
    @abstractmethod
    def loss_type(self):
        pass

    @property
    @abstractmethod
    def codec_ctrl(self):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def target_acc(self):
        pass

    @property
    @abstractmethod
    def psgd_type(self):
        pass

    @property
    @abstractmethod
    def psgd_server_codec(self):
        pass

    @property
    @abstractmethod
    def psgd_server_type(self):
        pass

    @property
    @abstractmethod
    def optimizer_type(self):
        pass

    @property
    @abstractmethod
    def epoches(self):
        pass

    @property
    @abstractmethod
    def learn_rate(self):
        pass


