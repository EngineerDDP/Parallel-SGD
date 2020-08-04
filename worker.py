import time
import os

from psgd.psgd_training_client import PSGDTraining_Client
from utils.constants import Initialization_Server
from codec.tag import Tag
from utils.log import Logger
from network.communications import Communication_Controller, Worker_Communication_Constructor, get_repr
from utils.models import *

# network agreements in used
from network.starnet_com_process import Worker_Register, Communication_Process, STAR_NET_WORKING_PORTS


CLZ_WORKREGISTER = Worker_Register
CLZ_COM_PROCESS = Communication_Process


def build_tags(node_id: int):
    tags = []

    for block in GlobalSettings.get_default().block_assignment.node_2_block[node_id]:
        tags.append(Tag(GlobalSettings.get_default().batch,
                        block,
                        node_id,
                        set(GlobalSettings.get_default().block_assignment.block_2_node[block])))

    return tags


class PSGD_Worker:

    def __init__(self):
        self.__running_thread = None
        self.client_logger = Logger(title_info='Worker-{}'.format(get_repr()), log_to_file=True)
        self.__training_log = None

        self.client_logger.log_message('Working started and ready for job submission.')

    def slave_forever(self):
        while True:
            constructor = Worker_Communication_Constructor('0.0.0.0', STAR_NET_WORKING_PORTS, worker_register=Worker_Register())
            try:
                register = constructor.buildCom()
                com = Communication_Controller(Communication_Process(register))

                self.client_logger.log_message('Job submission received. Node assigned node_id({})'.format(com.Node_ID))

                self.init_PSGD(com)
                self.do_training(com)
                # wait for safe closure
                time.sleep(10)
                com.close()
            except Exception as e:
                self.client_logger.log_message('Exception occurred: {}'.format(e))

    def init_PSGD(self, com: Communication_Controller) -> bool:
        # initialize global settings
        com.send_one(Initialization_Server, Init.GlobalSettings)
        _, data = com.get_one()
        # restore global settings
        if not isinstance(data, Reply.global_setting_package):
            if isinstance(data, Reply):
                if data == Reply.I_Need_Your_Working_Log:
                    com.send_one(Initialization_Server, Binary_File_Package(self.client_logger.File_Name))
            return False

        try:
            data.restore()

            # initialize codec and sgd type
            com.send_one(Initialization_Server, Init.Codec_And_SGD_Type)
            _, data = com.get_one()
            # restore
            assert isinstance(data, Reply.codec_and_sgd_package)

            codec, sgd = data.restore()

            # initialize weights and layer
            com.send_one(Initialization_Server, Init.Weights_And_Layers)
            _, data = com.get_one()
            # restore
            assert isinstance(data, Reply.weights_and_layers_package)

            layers = data.restore()

            # initialize dataset
            com.send_one(Initialization_Server, Init.Samples)
            _, data = com.get_one()
            # restore
            assert isinstance(data, Reply.data_sample_package)

            train_x, train_y, eval_x, eval_y = data.restore()

            # others
            com.send_one(Initialization_Server, Init.MISC)
            _, data = com.get_one()
            assert isinstance(data, Reply.misc_package)

            loss_t = data.loss_type
            target_acc = data.target_acc
            epoch = data.epoch
            learn_rate = data.learn_rate
            w_type = data.w_types

            self.__training_log = Logger('Training log @ node-{}'.format(com.Node_ID), log_to_file=True)

            self.__running_thread = PSGDTraining_Client(
                model_init=layers,
                loss=loss_t,
                codec_type=codec,
                sync_class=sgd,
                com=com,
                w_types=w_type,
                tags=build_tags(node_id=com.Node_ID),
                train_x=train_x,
                train_y=train_y,
                eval_x=eval_x,
                eval_y=eval_y,
                batch_size=GlobalSettings.get_default().batch.batch_size,
                epochs=epoch,
                logger=self.__training_log,
                learn_rate=learn_rate,
                target_acc=target_acc
            )

            return True
        except Exception as error:
            self.client_logger.log_message('Error encountered while initializing training environment : {}.'.format(error))
            return False

    def do_training(self, com: Communication_Controller):
        self.client_logger.log_message('Prepare to start training process.')
        # check
        assert isinstance(self.__running_thread, PSGDTraining_Client)
        assert isinstance(self.__training_log, Logger)

        ready_state = {}
        self.client_logger.log_message('Synchronize timeline with cluster.')
        self.__training_log.log_message(self.__running_thread.Model.summary())

        for node_id in GlobalSettings.get_default().nodes:
            com.send_one(node_id, Ready_Type())

        # check ready states
        while len(ready_state) != GlobalSettings.get_default().node_count:
            time.sleep(0.1)
            # require
            n, d = com.get_one()
            if isinstance(d, Ready_Type):
                ready_state[n] = True

        # make output file
        if not os.path.exists('./training'):
            os.mkdir('./training')
        try:
            begin = time.time()
            self.__running_thread.start()
            self.__running_thread.join()
            end = time.time()

            self.__training_log.log_message('Execution complete, time:{}'.format(end - begin))
            self.client_logger.log_message('Execution complete, time:{}'.format(end - begin))

            train_csv = Binary_File_Package(self.__running_thread.Trace_Train)
            eval_csv = Binary_File_Package(self.__running_thread.Trace_Eval)

            com.send_one(Initialization_Server, train_csv)
            com.send_one(Initialization_Server, eval_csv)
        except Exception as error:
            self.client_logger.log_message('Error encountered while executing : {}'.format(error))
            self.__training_log.log_message('Error encountered while executing : {}'.format(error))

        self.client_logger.log_message('Training process exited.')
        log_file = Binary_File_Package(self.__training_log.File_Name)
        com.send_one(Initialization_Server, log_file)


