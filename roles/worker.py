import time

from executor.interfaces import IExecutor

from models.trans import IReplyPackage, RequestWorkingLog, Binary_File_Package, Done_Type, Ready_Type, SubmitJob

from network.communications import get_repr
from network import ICommunication_Controller, Serve

from utils.constants import Initialization_Server
from utils.log import Logger


class PSGD_Worker:

    def __init__(self):
        self.client_logger = Logger(title_info='Worker-{}'.format(get_repr()), log_to_file=True)
        self.client_logger.log_message('Working started and ready for job submission.')
        self.__job_executor = None

    def slave_forever(self):
        # set up listening port
        listener = Serve(net_type='fcnet')
        try:
            while True:
                self.client_logger.log_message('Worker started with network type \'FCNet\'.')
                try:
                    with listener.acquire() as com:
                        self.client_logger.log_message('Job submission received. Node assigned node_id({})'.format(com.Node_Id))

                        self.dispatch(com)

                        self.client_logger.log_message('Current session closed, node_id({}).'.format(com.Node_Id))
                        self.client_logger.log_message('Worker restarting...')
                        time.sleep(1)
                except OSError:
                    self.client_logger.log_message("Initialization server exited without report.")
                except ConnectionResetError:
                    self.client_logger.log_message("Initialization server exited without report.")

        except KeyboardInterrupt:
            self.client_logger.log_error('Worker shutdown by interruption.')
            listener.close()

    @staticmethod
    def __recv_pack(com:ICommunication_Controller, timeout:int=100):
        data = None
        id_from = None
        time_clock = 0
        # requests with timeout check
        while data is None:
            id_from, data = com.get_one(blocking=False)
            time.sleep(0.001)
            time_clock += 0.002
            # Assertion, this node count as one
            assert Initialization_Server in com.available_clients, "Initialization server exited without finishing the initialization."
            assert time_clock < timeout, "Maximum waiting time exceed."
        return id_from, data

    def dispatch(self, com: ICommunication_Controller):
        """
            Get first package and find out what to do.
            All exceptions will be handled here, and trace back information will
            be recorded to client_logger.
            Used job_submit.py --retrieve to get trace back log.
        :param com:
        :return:
        """
        try:
            _, req = PSGD_Worker.__recv_pack(com, 7)
            if isinstance(req, SubmitJob):
                if self.init_PSGD(com, req):
                    self.do_training(com)

            if isinstance(req, RequestWorkingLog):
                self.post_log(com)

            com.send_one(Initialization_Server, Done_Type())
        except Exception as e:
            self.client_logger.log_error('Exception occurred: {}'.format(e))

            # print DEBUG message
            import sys
            import traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            for line in exc_tb:
                self.client_logger.log_error(line[:-1])
            # print DEBUG message

    def post_log(self, com: ICommunication_Controller):
        """
            Post worker log file to coordinator.
        :param com:
        :return:
        """
        self.client_logger.log_message('Nothing needs to be done, send back logfile and exit process.')
        com.send_one(Initialization_Server, Binary_File_Package(self.client_logger.File_Name))
        if isinstance(self.__job_executor, IExecutor):
            for filename in self.__job_executor.trace_files():
                com.send_one(Initialization_Server, Binary_File_Package(filename))

    def init_PSGD(self, com: ICommunication_Controller, job_info: SubmitJob) -> bool:
        """
            Initialize P-SGD Training environment
        :param com: Communication process
        :param job_info: job info
        :return:
        """
        self.client_logger.log_message('Request (settings), (weights and layer).')
        # restoring data
        job_info.restore()
        # get info
        ready_state = set()
        total_nodes = job_info.work_group
        eta_waiting_time = job_info.waiting_time

        self.__job_executor: IExecutor = job_info.executioner(com.Node_Id, job_info.group_offset)

        # Acknowledge requests
        requests = self.__job_executor.requests()
        replies = []
        # Ask for replies
        for req in requests:
            com.send_one(Initialization_Server, req)

        # Set job executor to ready state
        while not self.__job_executor.ready():

            id_from, data = PSGD_Worker.__recv_pack(com, eta_waiting_time)

            self.client_logger.log_message('Ack package, type: ({})'.format(data.__class__.__name__))
            # restoring data
            if isinstance(data, IReplyPackage):
                data.restore()
                replies.append(data)

                if len(replies) == len(requests):
                    for req in self.__job_executor.satisfy(replies):
                        com.send_one(Initialization_Server, req)

            # pass to sync
            elif isinstance(data, Ready_Type):
                ready_state = ready_state | data.current_ready()

        self.client_logger.log_message('Submit stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.client_logger.log_message('Submit stage complete, Total bytes read: {}'.format(com.Com.bytes_read))

        self.client_logger.log_message('Synchronize timeline with cluster.')

        PSGD_Worker.synchronize(com, ready_state, total_nodes, eta_waiting_time)

        return True

    @staticmethod
    def synchronize(com:ICommunication_Controller, ready_state: set, total_nodes: set, timeout:int):
        """
            Synchronize timeline with cluster.
            Make sure all nodes exits this method with same time.
        :param com: communication controller
        :param ready_state: how much nodes is ready now
        :param total_nodes: how much nodes we need for the job
        :param timeout: timeout limit in seconds, vaguely accuracy
        :return:
        """
        timeout_clock = 0

        ready_state.add(com.Node_Id)
        for id in com.available_clients:
            com.send_one(id, Ready_Type(ready_state))

        while ready_state & total_nodes != total_nodes:
            assert timeout_clock < timeout, "Maximum waiting time exceed."

            current_active = set(com.available_clients) | {com.Node_Id}
            assert current_active & total_nodes == total_nodes, "Minimum nodes cannot be satisfied."
            # inc time clock
            time.sleep(1)
            timeout_clock += 1

            # check ready state
            id_from, data = com.get_one(blocking=False)
            if isinstance(data, Ready_Type):
                ready_state = ready_state | data.current_ready()

    def do_training(self, com: ICommunication_Controller):
        """
            Execute job.
        """
        self.client_logger.log_message('Execution process started.')
        begin = time.time()
        self.__job_executor.run(com)
        end = time.time()

        self.client_logger.log_message('Execution complete, time:{}'.format(end - begin))
        self.client_logger.log_message('Training stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.client_logger.log_message('Training stage complete, Total bytes read: {}'.format(com.Com.bytes_read))

        for filename in self.__job_executor.trace_files():
            data = Binary_File_Package(filename)
            com.send_one(Initialization_Server, data)

        self.client_logger.log_message('Try post training log to coordinator.')

        self.client_logger.log_message('Training process exited.')
