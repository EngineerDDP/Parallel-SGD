import time
import os
from threading import Thread

from psgd.psgd_training_client import PSGDTraining_Client, PSGDTraining_Parameter_Server
from utils.constants import Initialization_Server, Parameter_Server
from codec.tag import Tag
from utils.log import Logger
from network.communications import Communication_Controller, Worker_Communication_Constructor, get_repr
from utils.models import *

# network agreements in used
from network.starnet_com_process import Worker_Register, Communication_Process, STAR_NET_WORKING_PORTS


CLZ_WORKER_REGISTER = Worker_Register
CLZ_COM_PROCESS = Communication_Process


def build_tags(node_id: int):
    if not isinstance(node_id, int):
        node_id = int(node_id)

    assert node_id < GlobalSettings.get_default().node_count, "This worker has nothing to do."

    batch = GlobalSettings.get_default().batch
    blocks = GlobalSettings.get_default().block_assignment.node_2_block[int(node_id)]
    nodes = GlobalSettings.get_default().block_assignment.block_2_node
    tags = [Tag(batch, block, node_id, set(nodes[block])) for block in blocks]

    return tags


class PSGD_Worker:

    def __init__(self):
        self.__running_thread = None
        self.client_logger = Logger(title_info='Worker-{}'.format(get_repr()), log_to_file=True)
        self.__training_log = None

        self.client_logger.log_message('Working started and ready for job submission.')

    def slave_forever(self):
        # set up listening port
        constructor = Worker_Communication_Constructor('0.0.0.0', STAR_NET_WORKING_PORTS, worker_register=CLZ_WORKER_REGISTER())
        while True:
            com = None
            try:
                self.client_logger.log_message('Worker started, prepare for connection...')
                register = constructor.buildCom()
                com = Communication_Controller(CLZ_COM_PROCESS(register))
                com.establish_communication()

                self.client_logger.log_message('Job submission received. Node assigned node_id({})'.format(com.Node_Id))

                if self.init_PSGD(com):
                    self.do_training(com)

                GlobalSettings.clear_default()
                self.client_logger.log_message('Current session closed, node_id({}).'.format(com.Node_Id))

            except Exception as e:
                self.client_logger.log_error('Exception occurred: {}'.format(e))

                # print DEBUG message
                import sys
                import traceback
                exc_type, exc_value, exc_tb = sys.exc_info()
                exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
                for line in exc_tb:
                    self.client_logger.log_message(line)
                # print DEBUG message

            except KeyboardInterrupt:
                self.client_logger.log_error('Worker shutdown by interruption.')
                constructor.close()
                break
            finally:
                time.sleep(10)
                if isinstance(com, Communication_Controller):
                    com.close()

            self.client_logger.log_message('Worker restarting...')
            # wait for safe closure

    def init_PSGD(self, com: Communication_Controller) -> bool:
        self.client_logger.log_message('ACK job submission and request global settings.')
        # ignore other data
        def acquire(com):
            id_from, data = com.get_one()
            while id_from != Initialization_Server:
                id_from, data = com.get_one()
            return data
        # initialize global settings
        com.send_one(Initialization_Server, Init.GlobalSettings)
        # get data
        data = acquire(com)
        # restore global settings
        if not isinstance(data, Reply.global_setting_package):
            if data == Reply.I_Need_Your_Working_Log:
                self.client_logger.log_message('Nothing needs to be done, send back logfile and exit process.')
                com.send_one(Initialization_Server, Binary_File_Package(self.client_logger.File_Name))
                if isinstance(self.__training_log, Logger):
                    com.send_one(Initialization_Server, Binary_File_Package(self.__training_log.File_Name))
                if isinstance(self.__running_thread, PSGDTraining_Client):
                    com.send_one(Initialization_Server, Binary_File_Package(self.__running_thread.Trace_Eval))
                    com.send_one(Initialization_Server, Binary_File_Package(self.__running_thread.Trace_Train))
                com.send_one(Initialization_Server, Done_Type())
            return False

        data.restore()

        self.client_logger.log_message('Request codec and sgd class.')
        # initialize codec and sgd type
        com.send_one(Initialization_Server, Init.Codec_And_SGD_Type)

        data = acquire(com)
        assert isinstance(data, Reply.codec_and_sgd_package)

        codec, sgd = data.restore()

        self.client_logger.log_message('Request weights and layer type.')
        # initialize weights and layer
        com.send_one(Initialization_Server, Init.Weights_And_Layers)
        data = acquire(com)
        assert isinstance(data, Reply.weights_and_layers_package)

        layers = data.restore()

        self.client_logger.log_message('Request other stuff.')
        # others
        com.send_one(Initialization_Server, Init.MISC)
        data = acquire(com)
        assert isinstance(data, Reply.misc_package)

        loss_t = data.loss_type
        target_acc = data.target_acc
        epoch = data.epoch
        learn_rate = data.learn_rate
        w_type = data.w_types
        op = data.optimizer

        self.__training_log = Logger('Training log @ node-{}'.format(com.Node_Id), log_to_file=True)

        if com.Node_Id != Parameter_Server:

            self.client_logger.log_message('Request data samples.')
            # initialize dataset
            com.send_one(Initialization_Server, Init.Samples)
            data = acquire(com)
            # restore
            assert isinstance(data, Reply.data_sample_package)

            train_x, train_y, eval_x, eval_y = data.restore()

            self.__running_thread = PSGDTraining_Client(
                model_init=layers,
                loss=loss_t,
                codec_type=codec,
                sync_class=sgd,
                com=com,
                w_types=w_type,
                tags=build_tags(node_id=com.Node_Id),
                train_x=train_x,
                train_y=train_y,
                eval_x=eval_x,
                eval_y=eval_y,
                optimizer=op,
                batch_size=GlobalSettings.get_default().batch.batch_size,
                epochs=epoch,
                logger=self.__training_log,
                learn_rate=learn_rate,
                target_acc=target_acc
            )
        else:
            self.__running_thread = PSGDTraining_Parameter_Server(
                model_init=layers,
                ps_codec=codec,
                ps_sgd_type=sgd,
                com=com,
                w_types=w_type,
                logger=self.__training_log
            )

        self.client_logger.log_message('Submit stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
        self.client_logger.log_message('Submit stage complete, Total bytes read: {}'.format(com.Com.bytes_read))
        return True

    def do_training(self, com: Communication_Controller):
        self.client_logger.log_message('Prepare to start training process.')
        # check
        assert isinstance(self.__running_thread, Thread)
        assert isinstance(self.__training_log, Logger)

        ready_state = {}
        self.client_logger.log_message('Synchronize timeline with cluster.')

        len_ready = len(com.available_clients())

        # check ready states
        while len(ready_state) != len_ready:
            # require
            n, d = com.get_one(False)
            if isinstance(d, Ready_Type):
                ready_state[n] = True
            elif len(com.available_clients()) < len_ready:
                raise OSError('Minimal number of clients cannot be satisfied.')
            for node_id in com.available_clients():
                com.send_one(node_id, Ready_Type())
            time.sleep(1)

        # make output file
        if not os.path.exists('./training'):
            os.mkdir('./training')
        try:
            self.client_logger.log_message('Execution process started.')
            data_sent_mark = com.Com.bytes_sent
            data_recv_mark = com.Com.bytes_read
            begin = time.time()
            self.__running_thread.start()
            self.__running_thread.join()
            end = time.time()

            self.__training_log.log_message('Execution complete, time:{}'.format(end - begin))
            self.__training_log.log_message('Bytes sent: {}'.format(com.Com.bytes_sent - data_sent_mark))
            self.__training_log.log_message('Bytes read: {}'.format(com.Com.bytes_read - data_recv_mark))

            self.client_logger.log_message('Execution complete, time:{}'.format(end - begin))
            self.client_logger.log_message('Training stage complete, Total bytes sent: {}'.format(com.Com.bytes_sent))
            self.client_logger.log_message('Training stage complete, Total bytes read: {}'.format(com.Com.bytes_read))

            if isinstance(self.__running_thread, PSGDTraining_Client):
                train_csv = Binary_File_Package(self.__running_thread.Trace_Train)
                eval_csv = Binary_File_Package(self.__running_thread.Trace_Eval)

                self.client_logger.log_message('Post training log.')
                com.send_one(Initialization_Server, train_csv)
                com.send_one(Initialization_Server, eval_csv)

        except Exception as error:
            self.client_logger.log_error('Error encountered while executing : {}'.format(error))
            self.__training_log.log_error('Error encountered while executing : {}'.format(error))

        self.client_logger.log_message('Training process exited.')
        log_file = Binary_File_Package(self.__training_log.File_Name)
        com.send_one(Initialization_Server, log_file)


if __name__ == '__main__':
    PSGD_Worker().slave_forever()