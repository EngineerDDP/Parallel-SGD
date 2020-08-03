from abc import ABCMeta, abstractmethod
from multiprocessing import Value, Process
from multiprocessing import Queue


class IWorker_Register(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def register(self, id_self, content_package):
        """
            Register all workers
        :param id_self: id of current worker
        :param content_package: content package that contains address and uuid of all workers
        :return: None
        """
        pass

    @abstractmethod
    def check(self):
        pass

    @abstractmethod
    def put(self, id, con):
        pass

    @abstractmethod
    def identify(self, id, uuid, con):
        pass


class ICommunication_Process(Process, metaclass=ABCMeta):
    Circle_interval = 0.001

    def __init__(self, name: str):
        Process.__init__(self, name=name)
        self.Exit = Value('i', 0)
        self.recv_que = Queue()
        self.send_que = Queue()

    @abstractmethod
    def node_id(self):
        pass

    def close(self):
        self.Exit.value = True