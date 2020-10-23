from time import sleep

from codec import GlobalSettings
from executor.abstract import AbsExecutor
from executor.psgd.net_package import Req, net_setting
from models import RequestPackage
from network import ICommunication_Controller
from psgd.interface import ITransfer
from utils.log import Logger


class PSGDPSExecutor(AbsExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        # wait
        self.__log = Logger('ParaServer'.format(node_id), log_to_file=True)
        self.__done: [bool] = False
        self.__transfer: [ITransfer] = None

    def requests(self):
        return [Req.Setting, Req.Transfer_PS]

    def satisfy(self, reply:list) -> list:
        # check list
        for obj in reply:

            if isinstance(obj, net_setting):
                GlobalSettings.deprecated_default_settings = obj.setting()

            if isinstance(obj, ITransfer):
                self.__transfer = obj
                self.__log.log_message('Transfer thread is ready.')

        return []

    def ready(self) -> bool:
        return self.__transfer is not None \
                and GlobalSettings.deprecated_default_settings is not None

    def start(self, com: ICommunication_Controller) -> None:
        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        GlobalSettings.deprecated_global_logger = self.__log
        self.__transfer.start_transfer(com, printer=self.__log, group_offset=0)

        from utils.constants import Initialization_Server
        while set(com.available_clients) - {Initialization_Server} != set():
            sleep(7)

        data_sent_end = com.Com.bytes_sent
        data_recv_end = com.Com.bytes_read
        self.__log.log_message('Execution complete, Total bytes sent: {}.'.format(data_sent_end - data_send_start))
        self.__log.log_message('Execution complete, Total bytes read: {}.'.format(data_recv_end - data_recv_start))

    def trace_files(self) -> list:
        return [self.__log.File_Name]

    def done(self) -> bool:
        return self.__done
