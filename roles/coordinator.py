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
        self.__allocation_list = set()

    def resources_dispatch(self, settings:Settings, hyper_model: IServerModel, data_set:AbsDataset, data_trans:ITransformer):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """
        # dispatch to certain group
        node_ready = set()

        while node_ready == self.__allocation_list:

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
                    self.__log.log_message('Node({}) is ready, {} is ready.'.format(id_from, node_ready))

                self.__com.send_one(id_from, reply)

            except KeyboardInterrupt:
                if len(node_ready) < len(self.__allocation_list):
                    self.__log.log_error('Some of workers is not ready.')
                self.__log.log_error('Coordinator closed by user.')

        self.__log.log_message('Dispatch complete.')

    def join(self) -> None:
        """
            Join all workers, wait for all task.
        """
        # dispatch to certain group
        node_ready = set()

        self.__log.log_message("Waiting for ({}) ...".format(self.__allocation_list))
        while node_ready == self.__allocation_list:

            id_from, data = self.__com.get_one()

            if isinstance(data, Binary_File_Package):
                data.restore()
                self.__log.log_message('Restoring data ({}) from {}.'.format(data.filename, id_from))

            elif isinstance(data, Done_Type):
                node_ready.add(id_from)
                self.__log.log_message('Node({}) is done, {} is done.'.format(id_from, node_ready))

        self.__log.log_message("All task is complete.")

    def submit_job(self, worker_executor:type, worker_offset:int=0, worker_cnt:int=0, data_size:int=0, ps_executor:type=None):
        """
            Submit a job to cluster
        :return:
        """
        # set work group
        if worker_cnt == 0:
            working_group = set(self.__com.available_clients)
        else:
            working_group = set(range(worker_offset, worker_offset + worker_cnt))
        # check for duplication
        assert len(self.__allocation_list & working_group) == 0, "Cannot submit a task to node which already has a job."
        # calculate data size
        dataset_ett = self.__com.available_clients_count * data_size / Estimate_Bandwidth
        # send request
        for id in working_group:
            if id == Parameter_Server and ps_executor is not None:
                self.__com.send_one(id, SubmitJob(working_group, worker_offset, True, dataset_ett * worker_cnt, ps_executor))
            else:
                self.__com.send_one(id, SubmitJob(working_group, worker_offset, False, dataset_ett, worker_executor))
        self.__log.log_message("Submission complete.")


class Reclaimer:

    def __init__(self, com:ICommunication_Controller, logger:Logger=None):
        self.__com = com
        if logger is None:
            self.__log = Logger(title_info='Retrieve', log_to_file=True)
        else:
            self.__log = logger

    def require_client_log(self):
        """
            Require client_log file from all workers.
        :return: None
        """
        # send request
        for id in self.__com.available_clients:
            self.__com.send_one(id, RequestWorkingLog())

        try:
            # get result
            for id in self.__com.available_clients:
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
