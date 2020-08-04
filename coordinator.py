from utils.log import Logger
from utils.models import *
from utils.constants import *

from network.communications import Communication_Controller, get_repr
from server_util.init_model import IServerModel

# method to start up a network
from network.starnet_com_process import start_star_net, StarNetwork_Initialization_Package
NET = start_star_net
IPA = StarNetwork_Initialization_Package

class Coordinator:

    def __init__(self, hyper_model: IServerModel):
        self.__com = None
        self.__model = hyper_model
        self.__log = Logger(title_info='Coordinator-{}'.format(get_repr()), log_to_file=True)

    def set_workers(self, works: list) -> None:
        """
            Set worker list.
        :param works: list of tuples
                        like: [ (id1, uuid1, address1), (id2, uuid2, address2), ... ]
        :return: None, raise exceptions if two workers with same id are assigned.
        """
        pkg = IPA()
        # set all address
        for id, uuid, addr in works:
            pkg.put(id, uuid, addr)
            self.__log.log_message('Add worker (id: {}, address: {}).'.format(id, addr))

        self.__com = NET(pkg)
        self.__com = Communication_Controller(self.__com)

    def resources_dispatch(self):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """
        # assertion
        assert isinstance(self.__com, Communication_Controller)
        assert isinstance(self.__model, IServerModel)

        while not self.__com.is_closed():
            id_from, data = self.__com.get_one()

            if isinstance(data, Init):
                if data == Init.GlobalSettings:
                    reply = Reply.global_setting_package(GlobalSettings.get_default())

                elif data == Init.Weights_And_Layers:
                    reply = Reply.weights_and_layers_package(self.__model.getWeightsInit())

                elif data == Init.Codec_And_SGD_Type:
                    if id_from != Parameter_Server:
                        reply = Reply.codec_and_sgd_package(
                            self.__model.codec_ctrl(),
                            self.__model.psgd_type()
                        )
                    else:
                        reply = Reply.codec_and_sgd_packages(
                            self.__model.psgd_server_codec(),
                            self.__model.psgd_server_type()
                        )

                elif data == Init.Samples:
                    reply = Reply.data_sample_package(*self.__model.train_data(), *self.__model.eval_data())

                elif data == Init.MISC:
                    reply = Reply.misc_package(
                        self.__model.epoches(),
                        self.__model.loss_type(),
                        self.__model.learn_rate(),
                        self.__model.target_acc(),
                        self.__model.weights_types(),
                        self.__model.optimizer_type()
                    )

                else:
                    reply = None

                self.__log.log_message('Reply requirements ({}) to {}'.format(id_from, data.__class__.__name__))
                self.__com.send_one(id_from, reply)

            elif isinstance(data, Binary_File_Package):
                data.restore()
                self.__log.log_message('Restoring data ({}) from {}.'.format(data.filename, id_from))


    def require_client_log(self):
        """
            Require client_log file from all workers.
        :return: None
        """
        assert isinstance(self.__com, Communication_Controller)

        # take all ACK
        for id in self.__com.available_clients():
            _, _ = self.__com.get_one()

        # send request
        for id in self.__com.available_clients():
            self.__com.send_one(id, Reply.I_Need_Your_Working_Log)

        # get result
        for id in self.__com.available_clients():
            _, log = self.__com.get_one()
            if isinstance(log, Binary_File_Package):
                log.restore()

        return