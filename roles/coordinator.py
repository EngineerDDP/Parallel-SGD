from utils.constants import Parameter_Server, Estimate_Bandwidth
from utils.log import Logger

from models.trans import *
from models.trans.net_package import *

from network.interfaces import ICommunication_Controller
from models.local.neural_models import IServerModel


class Coordinator:

    def __init__(self, com:ICommunication_Controller, settings:Settings, logger=None):
        self.__com = com
        self.__setting = settings
        if logger is None:
            self.__log = Logger(title_info='Coordinator', log_to_file=True)
        else:
            self.__log = logger

    def resources_dispatch(self, hyper_model: IServerModel, data_set:AbsDataset, data_trans:ITransformer):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """
        total_node_count = len(self.__com.available_clients())
        worker_node_count = self.__setting.node_count
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
                        reply = global_setting_package(self.__setting)

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

    def submit_job(self, estimate_data_size:int, worker_execution_cls:type, ps_execution_cls:type):
        """
            Submit a job to cluster
        :return:
        """
        # calculate data size
        total_nodes = len(self.__com.available_clients())
        dataset_ett = total_nodes * estimate_data_size / Estimate_Bandwidth
        # send request
        for id in self.__com.available_clients():
            if id == Parameter_Server:
                self.__com.send_one(id, SubmitJob(total_nodes, True, dataset_ett * total_nodes, ps_execution_cls))
            else:
                self.__com.send_one(id, SubmitJob(total_nodes, False, dataset_ett, worker_execution_cls))

    def require_client_log(self):
        """
            Require client_log file from all workers.
        :return: None
        """
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


def set_workers(register:type, build_com_func, works: dict, nodes_required, log:Logger=None) -> ICommunication_Controller:
    """
        Set worker list.
    :param works: list of tuples
                    like: [ (rule1, address1), (rule2, address2), ... ]
    :return: None, raise exceptions if two workers with same id are assigned.
    """
    import random
    from network.communications import Communication_Controller

    pkg = register()
    uuid_for_this_task = str(random.randint(0, 0x7fffffff))
    current_node_id_assigned = 0

    if works.get("PS") is not None:
        pkg.put(Parameter_Server, uuid_for_this_task, works["PS"])
        if log is not None:
            log.log_message("Add parameter server: address: ({})".format(works["PS"]))

    for addr in works["Worker"]:
        pkg.put(current_node_id_assigned, uuid_for_this_task, addr)
        if log is not None:
            log.log_message("Add worker: id: ({}), address: ({})".format(current_node_id_assigned, addr))
        current_node_id_assigned += 1
        if current_node_id_assigned >= nodes_required:
            break

    com = build_com_func(pkg)
    com = Communication_Controller(com)
    com.establish_communication()

    return com