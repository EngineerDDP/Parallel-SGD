from time import sleep

from codec import GlobalSettings
from executor.abstract import AbsExecutor
from executor.psgd.net_package import Req, net_setting, extra_package
from network import ICommunication_Controller
from psgd.interface import ITransfer
from utils.log import Logger


class PSGDPSExecutor(AbsExecutor):

    def __init__(self, node_id: int, working_group: set, initializer_id: int = -1):
        super().__init__(node_id, working_group, initializer_id)
        # wait
        self.__log = Logger('ParaServer'.format(node_id), log_to_file=True)
        self.__done: [bool] = False
        self.__transfer: [ITransfer] = None

    def requests(self):
        return [Req.Setting, Req.Extra_Content]

    def satisfy(self, reply: list) -> list:
        unsatisfied = []
        # check list
        for obj in reply:

            if isinstance(obj, net_setting):
                GlobalSettings.deprecated_default_settings = obj.setting()

            if isinstance(obj, extra_package):
                GlobalSettings.global_parameters = obj.acquire()
                unsatisfied.append(Req.Transfer_PS)

            if isinstance(obj, ITransfer):
                self.__transfer = obj
                self.__log.log_message('Transfer thread is ready.')

        return unsatisfied

    def ready(self) -> bool:
        return self.__transfer is not None \
                and GlobalSettings.deprecated_default_settings is not None

    def start(self, com: ICommunication_Controller) -> None:
        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        GlobalSettings.deprecated_global_logger = self.__log
        self.__transfer.start_transfer(com, printer=self.__log, group_offset=0)

        while set(com.available_clients) - {self.initializer_id} != set():
            sleep(7)

        data_sent_end = com.Com.bytes_sent
        data_recv_end = com.Com.bytes_read
        self.__log.log_message('Execution complete, Total bytes sent: {}.'.format(data_sent_end - data_send_start))
        self.__log.log_message('Execution complete, Total bytes read: {}.'.format(data_recv_end - data_recv_start))

    def trace_files(self) -> list:
        return [self.__log.File_Name]

    def done(self) -> bool:
        return self.__done
