from time import sleep

from codec import GlobalSettings
from executor.abstract import AbsExecutor
from models import IServerModel
from models.trans import Req
from network import ICommunication_Controller
from profiles import Settings
from psgd.transfer import NTransfer
from utils.log import Logger


class PSGDPSExecutor(AbsExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        # wait
        self.__log = Logger('ParaServer'.format(node_id), log_to_file=True)
        self.__essential : IServerModel = None
        self.__setting : Settings = None
        self.__done : bool = False

    def requests(self) -> list:
        return [Req.Weights_And_Layers, Req.GlobalSettings]

    def satisfy(self, reply:list) -> list:
        # check list
        for obj in reply:

            if isinstance(obj, Settings):
                self.__add_setting(obj)

            elif isinstance(obj, IServerModel):
                self.__add_info(obj)

        return []

    def __add_setting(self, obj:Settings):
        self.__setting = obj
        # register global settings
        GlobalSettings.deprecated_default_settings = obj

    def __add_info(self, obj:IServerModel):
        self.__essential = obj

    def ready(self) -> bool:
        return self.__essential is not None \
                and self.__setting is not None

    def start(self, com:ICommunication_Controller) -> None:
        # build weights updater
        sgd_type = self.__essential.psgd_server_type
        iterator = range(len(self.__essential.codec_ctrl))
        codec_type = self.__essential.psgd_server_codec
        weights_updater = [
            {w:sgd_type(self.node_id, i, codec_type) for w in self.__essential.weights_types} for i in iterator
        ]
        # build transfer thread
        transfer = NTransfer(weights_updater, com, 0, self.__log)
        self.__log.log_message('Transfer thread is ready.')

        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        transfer.start_transfer()

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
