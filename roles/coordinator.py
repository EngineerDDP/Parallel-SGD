from models.trans import SubmitJob, Binary_File_Package, Req, Ready_Type, Done_Type, RequestWorkingLog

from network.interfaces import ICommunication_Controller
from utils.log import Logger


class Coordinator:

    def __init__(self, com:ICommunication_Controller, estimate_bandwidth=180000, logger=None):
        self.__com = com
        if logger is None:
            self.__log = Logger(title_info='Coordinator', log_to_file=True)
        else:
            self.__log = logger
        self.__estimate_bandwidth = estimate_bandwidth
        self.__group_allocated = set()
        self.__global_allocated = set()

    @property
    def allocated_nodes(self):
        return self.__global_allocated | self.__group_allocated

    def resources_dispatch(self, dispatch_map):
        """
            Reply to worker's requirements, prepare for the job
        :return:
        """
        # dispatch to certain group
        node_ready = set()

        while node_ready != self.allocated_nodes:

            try:
                id_from, data = self.__com.get_one()
                reply = None

                if isinstance(data, Req):
                    reply = dispatch_map[data]

                    self.__log.log_message('Reply requirements to node({}), type({}).'.format(id_from, reply.__class__.__name__))

                elif isinstance(data, Ready_Type):
                    reply = Ready_Type(node_ready)

                    if id_from in node_ready:
                        continue

                    node_ready.add(id_from)
                    self.__log.log_message('Node({}) is ready, {} is ready.'.format(id_from, node_ready))

                self.__com.send_one(id_from, reply)

            except KeyboardInterrupt:
                if len(node_ready) < len(self.allocated_nodes):
                    self.__log.log_error('Some workers are not ready.')
                self.__log.log_error('Coordinator closed by user.')

        self.__log.log_message('Dispatch complete.')

    def join(self) -> None:
        """
            Join all workers, wait for all task.
        """
        # dispatch to certain group
        node_ready = set()

        self.__log.log_message("Waiting for ({}) ...".format(self.allocated_nodes))

        while node_ready != self.allocated_nodes:

            id_from, data = self.__com.get_one()

            if isinstance(data, Binary_File_Package):
                data.restore()
                self.__log.log_message('Restoring data ({}) from {}.'.format(data.filename, id_from))

            elif isinstance(data, Done_Type):
                data.restore()
                self.__log.log_message('Restoring data from {}.'.format(id_from))
                node_ready.add(id_from)
                self.__log.log_message('Node({}) is done, {} is done.'.format(id_from, node_ready))

        self.__log.log_message("All task is complete.")

    def submit_group(self, worker_executor:type, worker_offset:int=0, worker_cnt:int=0, package_size:int=0):
        """
            Submit a job to a specified worker group.
            Nodes inside this group will wait for each other and synchronize start time.
            Group will also wait for all single nodes were ready.
        :param worker_executor: executor class, implementation of IExecutor
        :param worker_offset: Worker group offset. (Offset) is the first id in the group
        :param worker_cnt: Number of workers. (Offset + Worker_cnt - 1) is the last id in the group.
        :param package_size: Package size in transmission. Potentially required by executor, and provided by dispatch.
        :return: None
        """
        # set work group
        if worker_cnt == 0:
            working_group = set(self.__com.available_clients)
        else:
            working_group = set(range(worker_offset, worker_offset + worker_cnt))
        # check for duplication
        assert len(self.__group_allocated & working_group) == 0, "Cannot submit a task to node which already has a job."
        # calculate data size
        dataset_ett = self.__com.available_clients_count * package_size / self.__estimate_bandwidth
        # send request
        for id in working_group:
            self.__com.send_one(id, SubmitJob(working_group | self.__global_allocated, dataset_ett, worker_executor))

        self.__group_allocated = self.__group_allocated | working_group
        self.__log.log_message("Group submission complete ({}).".format(working_group))

    def submit_single(self, worker_executor:type, worker_id, package_size:int=0):
        """
            Submit a job to a specified node.
            This global node will start execution immediately when itself was ready.
        :param worker_executor: executor class, implementation of IExecutor
        :param worker_id: Worker id.
        :param package_size: Package size in transmission. Potentially required by executor, and provided by dispatch.
        :return:
        """
        # check for duplication
        assert worker_id not in self.__global_allocated, "Cannot submit a task to node which already has a job."
        # calculate data size
        dataset_ett = self.__com.available_clients_count * package_size / self.__estimate_bandwidth
        # send request
        self.__com.send_one(worker_id, SubmitJob({worker_id}, dataset_ett, worker_executor))

        self.__global_allocated.add(worker_id)
        self.__log.log_message("Single node submission complete.")


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
            self.__log.log_message('Acquire log file from worker({}).'.format(id))

        try:
            nodes_ready = set()
            total_nodes = set(self.__com.available_clients)
            while nodes_ready != total_nodes:
                id_from, log = self.__com.get_one()
                if isinstance(log, Binary_File_Package):
                    log.restore()
                    self.__log.log_message('Save log file for worker({}).'.format(id_from))
                elif isinstance(log, Done_Type):
                    nodes_ready.add(id_from)
                    log.restore()
                    self.__log.log_message('All package received for worker({})'.format(id_from))
        except:
            self.__log.log_error('Connection lost.')

        self.__log.log_message('Done.')
