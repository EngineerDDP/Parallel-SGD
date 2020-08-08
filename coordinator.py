from utils.log import Logger
from utils.models import *
from utils.constants import *

from network.communications import Communication_Controller, get_repr
from server_util.init_model import IServerModel

import random

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
                        like: [ (rule1, address1), (rule2, address2), ... ]
        :return: None, raise exceptions if two workers with same id are assigned.
        """
        pkg = IPA()
        uuid_for_this_task = str(random.randint(0, 0x7fffffff))
        current_node_id_assigned = 0
        # set all address
        for rule, addr in works:
            if rule == "PS":
                _id = Parameter_Server
            else:
                _id = current_node_id_assigned
                current_node_id_assigned += 1
            pkg.put(_id, uuid_for_this_task, addr)
            self.__log.log_message('Add worker (Rule: {}, Id: {}, Address: {}).'.format(rule, _id, addr))

        self.__log.log_message('Try connecting to the cluster.')
        self.__com = NET(pkg)
        self.__com = Communication_Controller(self.__com)
        self.__com.establish_communication()
        self.__log.log_message('Connection with cluster established.')

    def resources_dispatch(self):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """

        # assertion
        assert isinstance(self.__com, Communication_Controller)
        assert isinstance(self.__model, IServerModel)

        total_node_count = len(self.__com.available_clients())
        node_ready = set()
        key_interrupted_before = False

        while not self.__com.is_closed():
            try:
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
                            reply = Reply.codec_and_sgd_package(
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

                    self.__log.log_message('Reply requirements to node({}), type({}).'.format(id_from, reply.__class__.__name__))
                    self.__com.send_one(id_from, reply)

                elif isinstance(data, Ready_Type):
                    self.__com.send_one(id_from, Ready_Type())
                    if id_from in node_ready:
                        continue
                    node_ready.add(id_from)
                    self.__log.log_message('Node({}) is ready, {} nodes in total, {} is ready.'.format(id_from, total_node_count, node_ready))

                elif isinstance(data, Binary_File_Package):
                    self.__log.log_message('Restoring data ({}) from {}.'.format(data.filename, id_from))
                    data.restore()

            except KeyboardInterrupt:
                if len(node_ready) < total_node_count:
                    self.__log.log_error('Some of workers is not ready, close anyway?')
                    self.__log.log_message('Press Ctrl+C again to shutdown immediately.')
                    key_interrupted_before = True
                if key_interrupted_before or len(node_ready) >= total_node_count:
                    self.__log.log_error('Coordinator closed by user.')
                    break

        self.__com.close()
        self.__log.log_message('Dispatcher closed.')


    def require_client_log(self):
        """
            Require client_log file from all workers.
        :return: None
        """
        assert isinstance(self.__com, Communication_Controller)
        # self.__log.log_message('Acquire log file from each worker.')
        # take all ACK
        for id in self.__com.available_clients():
            _, _ = self.__com.get_one()

        # send request
        for id in self.__com.available_clients():
            self.__com.send_one(id, Reply.I_Need_Your_Working_Log)

        try:
            # get result
            for id in self.__com.available_clients():
                self.__log.log_message('Acquire log file from worker({}).'.format(id))
                log = None
                while not isinstance(log, Done_Type):
                    _, log = self.__com.get_one()
                    if isinstance(log, Binary_File_Package):
                        log.restore()
                        self.__log.log_message('Save log file for worker({}).'.format(id))
        except:
            self.__log.log_error('Connection lost.')

        self.__com.close()
        self.__log.log_message('Done.')

        return