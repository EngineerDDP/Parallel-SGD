from typing import Iterable, Callable, Dict

from models import *

from network.interfaces import ICommunication_Controller
from utils.log import IPrinter, Logger
from utils.constants import VERSION, Initialization_Server


class Coordinator:

    def __init__(self, com: ICommunication_Controller, estimate_bandwidth: int = 10, logger: IPrinter = None):
        """
            Coordinator
        :param com: Communication Thread
        :param estimate_bandwidth: bandwidth estimation, Bytes per second
        :param logger: IPrinter
        """
        self.__com = com
        if logger is None:
            self.__log = Logger(title_info='Coordinator', log_to_file=True)
        else:
            self.__log = logger
        self.__estimate_bandwidth = estimate_bandwidth
        self.__group_allocated = set()
        self.__global_allocated = set()
        self.__log.log_message("Coordinator version: {}.".format(VERSION))

    @property
    def allocated_nodes(self):
        return self.__global_allocated | self.__group_allocated

    def resources_dispatch(self, dispatch_map: Callable[[int, object], IReplyPackage]):
        """
            Reply to worker's requirements, prepare for the job
        :param dispatch_map: Callable object, receive a IRequestPackage instance and returns IReplyPackage instance
                            for reply.
        :return:
        """
        # dispatch to certain group
        node_ready = set()

        while node_ready != self.allocated_nodes:

            try:
                id_from, data = self.__com.get_one()
                reply = None

                if isinstance(data, IRequestPackage):
                    reply = dispatch_map(id_from, data.content())

                    self.__log.log_message(
                        'Reply requirements to node({}), type({}).'.format(id_from, reply.__class__.__name__))

                elif isinstance(data, ReadyType):
                    reply = ReadyType(node_ready)

                    if id_from in node_ready:
                        continue

                    node_ready.add(id_from)
                    self.__log.log_message('Node({}) is ready, {} is ready.'.format(id_from, node_ready))

                elif isinstance(data, Version):
                    reply = Version(Initialization_Server)

                    self.__log.log_message("{}".format(data))

                self.__com.send_one(id_from, reply)

            except KeyboardInterrupt:
                if len(node_ready) < len(self.allocated_nodes):
                    self.__log.log_error('Some workers are not ready.')
                self.__log.log_error('Coordinator closed by user.')

        self.__log.log_message('Dispatch complete.')

    def join(self) -> Dict[int, object]:
        """
            Join all workers, wait for all task.
            :return: Returns a dict, indicates what has been returned from executor on each worker.
        """
        # Join all nodes.
        node_ready = set()
        # Collect result.
        results: Dict[int, object] = {}

        self.__log.log_message("Waiting for ({}) ...".format(self.allocated_nodes))

        while node_ready != self.allocated_nodes:

            id_from, data = self.__com.get_one()

            if isinstance(data, IReplyPackage):
                data.restore()
                self.__log.log_message('Restoring data ({}) from {}.'.format(data, id_from))

            if isinstance(data, DoneType):
                file_format = "\n\t\t--> ".join([filename for filename in data.file_list])
                self.__log.log_message('Save file for {}.\n\tList:\n\t\t--> {}'.format(id_from, file_format))

                node_ready.add(id_from)
                self.__log.log_message('Node({}) is done, {} is done.'.format(id_from, node_ready))

                results[id_from] = data.result

        self.__log.log_message("All task is complete.")
        return results

    def submit_group(self, worker_executor: type, working_group: Iterable[int] = None, package_size: int = 1e9):
        """
            Submit a job to a specified worker group.
            Nodes inside this group will wait for each other and synchronize start time.
            Group will also wait for all single nodes were ready.
        :param worker_executor: executor class, implementation of IExecutor
        :param working_group: Worker group list, iterable object, contains id of each worker in the group.
        :param package_size: Package size in transmission. Potentially required by executor, and provided by dispatch.
        :return: None
        """
        # set work group
        if working_group is None:
            working_group = set(self.__com.available_clients)
        if not isinstance(working_group, set):
            working_group = set(working_group)
        # check for duplication
        assert len(self.__group_allocated & working_group) == 0, "Cannot submit a task to node which already has a job."
        # calculate data size
        dataset_ett = self.__com.available_clients_count * package_size / self.__estimate_bandwidth + 1
        # send request
        for _id in working_group:
            self.__com.send_one(_id, SubmitJob(working_group | self.__global_allocated, dataset_ett, worker_executor))

        self.__group_allocated = self.__group_allocated | working_group
        self.__log.log_message("Group submission complete ({}).".format(working_group))

    def submit_single(self, worker_executor: type, worker_id: int, package_size: int = 1e9):
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
        dataset_ett = self.__com.available_clients_count * package_size / self.__estimate_bandwidth + 0.6
        # send request
        self.__com.send_one(worker_id, SubmitJob({worker_id}, dataset_ett, worker_executor))

        self.__global_allocated.add(worker_id)
        self.__log.log_message("Single node submission complete.")


class Reclaimer:

    def __init__(self, com: ICommunication_Controller, logger: Logger = None):
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

                if isinstance(log, DoneType):
                    log.restore()
                    file_format = "\n\t\t--> ".join([filename for filename in log.file_list])
                    self.__log.log_message('Save file for {}.\n\tList:\n\t\t--> {}'.format(id_from, file_format))
                    nodes_ready.add(id_from)
                    self.__log.log_message('Node({}) is done, {} is done.'.format(id_from, nodes_ready))

        except Exception as e:
            # print DEBUG message
            import sys
            import traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            exc_format = "".join(exc_tb)
            self.__log.log_error('Exception occurred: {}\n\t{}'.format(e, exc_format))
            # print DEBUG message

        self.__log.log_message('Done.')
