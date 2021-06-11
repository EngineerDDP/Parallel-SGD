import time

import constants
import executor
import models
import network
import utils.log
from network.communications import get_repr


class Worker:

    def __init__(self, logger: utils.log.IPrinter = None):
        if logger is None:
            self.__client_logger = utils.log.Logger(title_info='Worker-{}'.format(get_repr()), log_to_file=True)
        self.__client_logger.log_message('Worker version: {}.'.format(constants.VERSION))
        self.__job_executor: [executor.IExecutor] = None

    def slave_forever(self):
        # set up listening port
        listener = network.Serve(net_type='fcnet')
        try:
            while True:
                self.__client_logger.log_message('Worker started on port: {}'.format(constants.Network_Working_Ports))
                try:
                    with listener.acquire() as com:
                        self.__client_logger.log_message(
                            'Job submission received. Node assigned node_id({})'.format(com.Node_Id))

                        self.dispatch(com)

                        self.__client_logger.log_message('Current session closed, node_id({}).'.format(com.Node_Id))
                        self.__client_logger.log_message('Worker restarting...')
                        time.sleep(1)
                except OSError:
                    self.__client_logger.log_message("Initialization server exited without report.")
                except ConnectionResetError:
                    self.__client_logger.log_message("Initialization server exited without report.")

        except KeyboardInterrupt:
            self.__client_logger.log_error('Worker shutdown by interruption.')
            listener.close()

    @staticmethod
    def __recv_pack(com: network.ICommunication_Controller, timeout: int = 100):
        # requests with timeout check
        id_from, data = com.get_one(blocking=True, timeout=timeout)
        # Assertion, this node count as one
        assert constants.Initialization_Server in com.available_clients, "Initialization server exited without finishing the initialization."
        assert data is not None, "Maximum waiting time exceed."
        return id_from, data

    def dispatch(self, com: network.ICommunication_Controller):
        """
            Get first package and find out what to do.
            All exceptions will be handled here, and trace back information will
            be recorded to client_logger.
            Used job_submit.py --retrieve to get trace back log.
        :param com:
        :return:
        """
        results = None
        try:
            id_from = com.Node_Id
            req = None
            while id_from != constants.Initialization_Server:
                id_from, req = Worker.__recv_pack(com, constants.Init_Job_Submission_Timeout_Limit_Sec)

            if isinstance(req, models.SubmitJob):
                # Report Version
                com.send_one(constants.Initialization_Server, models.Version(node_id=com.Node_Id))
                self.__client_logger.log_message('ACK job submission.')
                if self.initialize(com, req):
                    results = self.do_training(com)

            if isinstance(req, models.RequestWorkingLog):
                self.__client_logger.log_message('ACK logfile reclaim.')

        except Exception as e:
            # print DEBUG message
            import sys
            import traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            exc_format = "".join(exc_tb)
            self.__client_logger.log_error('Exception occurred: {}\n\t{}'.format(e, exc_format))
            # print DEBUG message

        self.post_log(com, results)

    def post_log(self, com: network.ICommunication_Controller, other_contents: object):
        """
            Post worker log file to coordinator.
        :param other_contents: other content can be attached
        :param com:
        :return:
        """
        posting_files = []
        if self.__client_logger.ToFile:
            posting_files.append(self.__client_logger.File_Name)
        if isinstance(self.__job_executor, executor.abstract.AbsExecutor):
            for filename in self.__job_executor.trace_files():
                posting_files.append(filename)

        # Post files
        com.send_one(constants.Initialization_Server, models.DoneType(com.Node_Id, posting_files, other_contents))

    def initialize(self, com: network.ICommunication_Controller, job_info: models.SubmitJob) -> bool:
        """
            Initialize execution environment
        :param com: Communication process
        :param job_info: job info
        :return:
        """
        # restoring data
        job_info.restore()
        # get info
        ready_state = set()
        total_nodes = job_info.work_group
        eta_waiting_time = job_info.waiting_time

        self.__job_executor: executor.abstract.AbsExecutor = job_info.executioner(com.Node_Id, job_info.work_group)

        # Acknowledge requests
        requests = self.__job_executor.requests()
        replies = []
        # Ask for replies
        for req in requests:
            com.send_one(constants.Initialization_Server, models.RequestPackage(req))

        req_format = "\tRequests List:\n\t\t--> {}".format("\n\t\t--> ".join([str(req) for req in requests]))
        self.__client_logger.log_message('Request data: ({})\n{}'.format(len(requests), req_format))
        self.__client_logger.log_message('ETA: ({})'.format(eta_waiting_time))
        # Set job executor to ready state
        while not self.__job_executor.ready():

            id_from, data = Worker.__recv_pack(com, eta_waiting_time)

            self.__client_logger.log_message('Ack package, type: ({})'.format(data.__class__.__name__))
            # restoring data
            if isinstance(data, models.IReplyPackage):
                data.restore()
                replies.append(data)

                if len(replies) == len(requests):
                    requests = self.__job_executor.satisfy(replies)
                    for req in requests:
                        com.send_one(constants.Initialization_Server, models.RequestPackage(req))
                    self.__client_logger.log_message('Request data: ({}).'.format(requests))
                    self.__client_logger.log_message('ETA: ({})'.format(eta_waiting_time))
                    replies.clear()

            # pass to sync
            elif isinstance(data, models.ReadyType):
                ready_state = ready_state | data.current_ready()

        self.__client_logger.log_message('Submit stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.__client_logger.log_message('Submit stage complete, Total bytes read: {}'.format(com.Com.bytes_read))

        self.__client_logger.log_message('Synchronize timeline with cluster.')

        Worker.synchronize(com, ready_state, total_nodes, eta_waiting_time)

        return True

    @staticmethod
    def synchronize(com: network.ICommunication_Controller, ready_state: set, total_nodes: set, timeout: int):
        """
            Synchronize timeline with cluster.
            Make sure all nodes exits this method with same time.
        :param com: communication controller
        :param ready_state: how much nodes is ready now
        :param total_nodes: how much nodes we need for the job
        :param timeout: timeout limit in seconds, vaguely accuracy
        :return:
        """
        dead_line = time.time() + timeout

        ready_state.add(com.Node_Id)
        for id in com.available_clients:
            com.send_one(id, models.ReadyType(ready_state))

        while ready_state & total_nodes != total_nodes:
            assert time.time() < dead_line, "Maximum waiting time exceed."

            current_active = set(com.available_clients) | {com.Node_Id}
            assert current_active & total_nodes == total_nodes, \
                "Current nodes: {}, required nodes: {}.".format(current_active, total_nodes)
            # inc time clock
            time.sleep(0.01)

            # check ready state
            id_from, data = com.get_one(blocking=False)
            if isinstance(data, models.ReadyType):
                ready_state = ready_state | data.current_ready()

    def do_training(self, com: network.ICommunication_Controller) -> object:
        """
            Execute job.
        """
        self.__client_logger.log_message('Execution process started.')
        begin = time.time()
        result = self.__job_executor.start(com)
        end = time.time()

        self.__client_logger.log_message('Execution complete, time:{}'.format(end - begin))
        self.__client_logger.log_message('Execution stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.__client_logger.log_message('Execution stage complete, Total bytes read: {}'.format(com.Com.bytes_read))
        self.__client_logger.log_message('Execution process exited.')

        return result
