from utils.constants import Parameter_Server, Estimate_Bandwidth
from utils.log import Logger

from models.trans import *
from models.trans.net_package import *

from network.interfaces import ICommunication_Controller
from models.local.neural_models import IServerModel


class Coordinator:

    def __init__(self, com:ICommunication_Controller, logger=None):
        self.__com = com
        if logger is None:
            self.__log = Logger(title_info='Coordinator', log_to_file=True)
        else:
            self.__log = logger
        self.__request_before = False

    def resources_dispatch(self, settings:Settings, hyper_model: IServerModel, data_set:AbsDataset, data_trans:ITransformer):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """
        total_node_count = len(self.__com.available_clients())
        worker_node_count = settings.node_count
        node_ready = set()
        node_done = set()

        while len(node_done) < worker_node_count:

            try:
                id_from, data = self.__com.get_one()
                reply = None

                if isinstance(data, Req):
                    if data == Req.Weights_And_Layers:
                        reply = essentials(hyper_model)

                    elif data == Req.GlobalSettings:
                        reply = global_setting_package(settings)

                    elif data == Req.Dataset:
                        reply = data_package(data_set, data_trans)

                    elif data == Req.Samples:
                        reply = data_content(data_set, data_trans)

                    self.__log.log_message('Reply requirements to node({}), type({}).'.format(id_from, reply.__class__.__name__))

                elif isinstance(data, Ready_Type):
                    reply = Ready_Type(node_ready)

                    if id_from in node_ready:
                        continue

                    node_ready.add(id_from)
                    self.__log.log_message('Node({}) is ready, {} nodes total, {} is ready.'.format(id_from, total_node_count, node_ready))

                elif isinstance(data, Binary_File_Package):
                    data.restore()
                    self.__log.log_message('Restoring data ({}) from {}.'.format(data.filename, id_from))

                elif isinstance(data, Done_Type):
                    node_done.add(id_from)
                    self.__log.log_message('Node({}) is done, {} nodes total, {} is done.'.format(id_from, worker_node_count, node_done))

                self.__com.send_one(id_from, reply)

            except KeyboardInterrupt:
                if len(node_ready) < total_node_count:
                    self.__log.log_error('Some of workers is not ready.')
                self.__log.log_error('Coordinator closed by user.')

        self.__log.log_message('Dispatcher closed.')
        self.__com.close()

    def submit_job(self, worker_execution_cls:type, estimate_data_size:int=0, ps_execution_cls:type=None):
        """
            Submit a job to cluster
        :return:
        """
        assert self.__request_before is False, "Request can only use once."
        self.__request_before = True
        # calculate data size
        total_nodes = len(self.__com.available_clients())
        dataset_ett = total_nodes * estimate_data_size / Estimate_Bandwidth
        # send request
        for id in self.__com.available_clients():
            if id == Parameter_Server and ps_execution_cls is not None:
                self.__com.send_one(id, SubmitJob(total_nodes, True, dataset_ett * total_nodes, ps_execution_cls))
            else:
                self.__com.send_one(id, SubmitJob(total_nodes, False, dataset_ett, worker_execution_cls))

    def require_client_log(self):
        """
            Require client_log file from all workers.
        :return: None
        """
        assert self.__request_before is False, "Request can only use once."
        self.__request_before = True
        # send request
        for id in self.__com.available_clients():
            self.__com.send_one(id, RequestWorkingLog())

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

        self.__log.log_message('Done.')
        self.__com.close()
