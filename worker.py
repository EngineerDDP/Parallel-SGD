import time
import os

from client import Slave
from constants import Initialization_Server
from codec.tag import Tag
from log import Logger
from network.communications import Communication_Controller, Worker_Communication_Constructor
from profiles.settings import GlobalSettings

# network agreements in used
from network.starnet_com_process import Worker_Register, Communication_Process, STAR_NET_WORKING_PORTS


CLZ_WORKREGISTER = Worker_Register
CLZ_COM_PROCESS = Communication_Process

class Init:
    GlobalSettings = 'Req_GlobalSettings'
    Weights_And_Layers = 'Req_WeightsAndLayers'
    Codec_And_SGD_Type = 'Req_CodecAndSGD'
    Samples = 'Req_Samples'
    MISC = 'Req_OtherStuff'

class Reply:
    GlobalSettings = 'Reply_GlobalSettings'
    Weights_And_Layers = 'Reply_WeightsAndLayers'
    Codec_And_SGD_Type = 'Reply_CodecAndSGD'
    Samples = 'Reply_Samples'
    MISC = 'Reply_OtherStuff'

    class global_setting_package:

        def __init__(self, settings: GlobalSettings):
            self.n = settings.nodes
            self.r = settings.redundancy
            self.b = settings.batch.batch_size
            self.ass = settings.block_assignment.__class__

        def restore(self):
            GlobalSettings.set_default(self.n, self.r, self.b, self.ass)

    class weights_and_layers_package:

        def __init__(self, layers: list):
            self.__storage = [(i.__class__, i.params()) for i in layers]

        def restore(self):
            return [i[0](*i[1]) for i in self.__storage]

    class codec_and_sgd_package:

        def __init__(self, codec, sgd_t):
            self.__codec = codec
            self.__sgd_t = sgd_t

        def restore(self):
            return self.__codec, self.__sgd_t

    class data_sample_package:

        def __init__(self, train_x, train_y, test_x, test_y):
            self.__train_x = train_x
            self.__train_y = train_y
            self.__test_x = test_x
            self.__test_y = test_y

        def restore(self):
            return self.__train_x, self.__train_y, self.__test_x, self.__test_y

    class misc_package:

        def __init__(self, epoch, loss_type, learn_rate, target_acc, w_types):
            self.epoch = epoch
            self.loss_type = loss_type
            self.learn_rate = learn_rate
            self.target_acc = target_acc
            self.w_types = w_types

class Ready_Type:

    def __init__(self):
        pass

class Training_Doc:

    def __init__(self, filename):

        self.filename = filename
        self.content = b''
        with open(filename, 'rb') as f:
            self.content = f.read()

    def restore(self):
        with open(self.filename, 'wb') as f:
            f.write(self.content)


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
        self.client_logger = Logger(title_info='Client_log', log_to_file=True)
        self.__training_log = None

    def slave_forever(self):
        while True:
            constructor = Worker_Communication_Constructor('0.0.0.0', STAR_NET_WORKING_PORTS, worker_register=Worker_Register())
            try:
                register = constructor.buildCom()
                com = Communication_Controller(Communication_Process(register))
                self.init_PSGD(com)
                self.do_training(com)
                # wait for save close
                time.sleep(10)
                com.close()
            except Exception as e:
                self.client_logger.log_message('Exception occurred: {}'.format(e))


    def init_PSGD(self, com: Communication_Controller):
        # initialize global settings
        com.send_one(Initialization_Server, Init.GlobalSettings)
        _, data = com.get_one()
        # restore global settings
        assert isinstance(data, Reply.global_setting_package)

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

        self.__running_thread = Slave(
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

    def do_training(self, com: Communication_Controller):
        self.client_logger.log_message('Prepare to start training process.')
        # check
        assert isinstance(self.__running_thread, Slave)
        assert isinstance(self.__training_log, Logger)

        ready_state = {}
        self.client_logger.log_message('Synchronize timeline with cluster.')

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

        begin = time.time()
        self.__running_thread.start()
        self.__running_thread.join()
        end = time.time()

        self.__training_log.log_message('Execution complete, time:{}'.format(end - begin))
        self.client_logger.log_message('Execution complete, time:{}'.format(end - begin))

        log_file = Training_Doc(self.__training_log.File_Name)
        train_csv = Training_Doc(self.__running_thread.Trace_Train)
        eval_csv = Training_Doc(self.__running_thread.Trace_Eval)

        com.send_one(Initialization_Server, log_file)
        com.send_one(Initialization_Server, train_csv)
        com.send_one(Initialization_Server, eval_csv)


