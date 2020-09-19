from abc import ABCMeta, abstractmethod


class IExecutor(metaclass=ABCMeta):

    @abstractmethod
    def requests(self):
        """
            Requested types
        return: list of Req(Enum) objects.
        """
        pass

    @abstractmethod
    def satisfy(self, reply):
        """
            Satisfy requested data.
        :return: list of Req(Enum) contains requests which cannot be satisfied.
        """
        pass

    @abstractmethod
    def start(self, com):
        """
            Do the job.
        """
        pass

    @abstractmethod
    def ready(self):
        """
            Is the executor ready for the job.
        """
        pass

    @abstractmethod
    def done(self):
        """
            Is job done?
        """
        pass

    @abstractmethod
    def trace_files(self):
        """
            Return the filename list or executing trace.
        """
        pass
