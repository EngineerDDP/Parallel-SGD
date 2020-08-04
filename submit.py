from utils.models import *
from utils.constants import *

from network.communications import Communication_Controller
from server_util.init_model import IServerModel

# method to start up a network
from network.starnet_com_process import start_star_net, StarNetwork_Initialization_Package
NET = start_star_net
IPA = StarNetwork_Initialization_Package

class Coordinator:

    def __init__(self, hyper_model: IServerModel):
        self.__com = None
        self.__model = hyper_model

    def set_workers(self, works: list) -> None:
        pkg = IPA()
        # set all address
        for id, uuid, addr in works:
            pkg.put(id, uuid, addr)

        self.__com = NET(pkg)
        self.__com = Communication_Controller(self.__com)

    def resources_dispatch(self):
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

                self.__com.send_one(id_from, reply)

            elif isinstance(data, Binary_File_Package):
                data.restore()