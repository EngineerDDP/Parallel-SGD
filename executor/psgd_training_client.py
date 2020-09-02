import time
import pandas as pd

from time import sleep
from functools import singledispatch

from codec import build_tags, GlobalSettings
from executor.interfaces import IExecutor, Settings, IServerModel, IDataset, ICommunication_Controller
from models.trans import Req

from nn.model import SequentialModel_v2
from psgd.transfer import NTransfer

from utils.log import Logger


class PSGDWorkerExecutor(IExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        self.__log = Logger('Fit-{}'.format(node_id), log_to_file=True)
        self.__trace_filename = [self.__log.File_Name]
        # waiting for those
        self.__model : SequentialModel_v2 = None
        self.__setting : Settings = None
        self.__essential : IServerModel = None
        self.__data : IDataset = None
        self.__done : bool = False

    def requests(self) -> list:
        return [Req.Dataset, Req.GlobalSettings, Req.Weights_And_Layers]

    def satisfy(self, reply:list) -> list:
        unsatisfied = []
        # check list
        for obj in reply:

            if isinstance(obj, IDataset):
                if not obj.check():
                    unsatisfied.append(Req.Samples)
                else:
                    self.__add_data(obj)

            if isinstance(obj, Settings):
                self.__add_setting(obj)

            if isinstance(obj, IServerModel):
                self.__add_info(obj)

        return unsatisfied

    def __add_info(self, obj:IServerModel):
        self.__essential = obj
        self.__model = SequentialModel_v2(obj.get_nn(), self.__log)

    def __add_data(self, obj:IDataset):
        self.__data = obj

    def __add_setting(self, obj:Settings):
        self.__setting = obj
        # register global settings
        GlobalSettings.deprecated_default_settings = obj

    def ready(self) -> bool:
        return self.__model is not None \
               and self.__setting is not None \
               and self.__essential is not None \
               and self.__data is not None

    def done(self) -> bool:
        return self.__done

    def run(self, com:ICommunication_Controller) -> None:
        # build layer updater
        sgd_type = self.__essential.psgd_type
        iterator = range(len(self.__essential.codec_ctrl))
        codec_type = self.__essential.codec_ctrl
        weights_updater = [
            {w:sgd_type(self.node_id, i, codec_type[i]) for w in self.__essential.weights_types} for i in iterator
        ]
        # build transfer thread
        transfer = NTransfer(weights_updater, com, self.group_offset, self.__log)
        # get batch size
        batch_size = self.__setting.batch.batch_size
        # build tags
        tags = build_tags(self.node_id, self.__setting)
        # build optimizer
        optimizer = self.__essential.optimizer_type(tags=tags, batch_size=batch_size, com=transfer, learn_rate=self.__essential.learn_rate)
        # compile model
        self.__model.compile(optimizer=optimizer, loss=self.__essential.loss_type(), metrics=self.__essential.metric)
        # summary
        self.__model.summary()
        trace_head = 'N({})-R({})-ID({})-CODEC({})'.format(self.__setting.node_count, self.__setting.redundancy,
                                                           self.node_id, ''.join([cc.__name__[0] for cc in self.__essential.codec_ctrl]))
        self.__log.log_message('Model set to ready.')
        # get dataset
        train_x, train_y, test_x, test_y = self.__data.load()

        self.__log.log_message('Dataset is ready, type: ({})'.format(self.__data))
        log_head = self.__log.Title
        # start !
        transfer.start_transfer()
        # record data
        time_start = time.time()
        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        evaluation_history = []
        # do until reach the target accuracy
        for i in range(self.__essential.epoches):
            # change title
            self.__log.Title = log_head + "-Epo-{}".format(i + 1)
            self.__model.fit(train_x, train_y, epochs=1, batch_size=batch_size)
            # do tests
            r = self.__model.evaluate(test_x, test_y)
            self.__log.log_message('Evaluate result: {}'.format(dict(zip(self.__model.History_Title[-len(r):], r))))
            evaluation_history.append(r)

            if self.__essential.target_acc is not None:
                # only one metric in model metrics list.
                # evaluation[0] refers to loss
                # evaluation[1] refers to accuracy.
                if r[1] > self.__essential.target_acc:
                    break

        # record data
        time_end = time.time()
        data_sent_end = com.Com.bytes_sent
        data_recv_end = com.Com.bytes_read

        training_history = self.__model.History
        # save training history data
        training_name = "TR-" + trace_head + ".csv"
        training_trace = pd.DataFrame(training_history, columns=self.__model.History_Title)
        training_trace.to_csv(training_name, index=None)
        # save evaluation history data
        evaluation_name = "EV-" + trace_head + ".csv"
        evaluation_trace = pd.DataFrame(evaluation_history, columns=self.__model.Evaluation_Title)
        evaluation_trace.to_csv(evaluation_name, index=None)
        self.__trace_filename.append(training_name)
        self.__trace_filename.append(evaluation_name)

        self.__log.log_message('Execution complete, time: {}.'.format(time_end - time_start))
        self.__log.log_message('Execution complete, Total bytes sent: {}.'.format(data_sent_end - data_send_start))
        self.__log.log_message('Execution complete, Total bytes read: {}.'.format(data_recv_end - data_recv_start))
        self.__log.log_message('Trace file has been saved to {}.'.format(trace_head))

        # set marker
        self.__done = True
        # dispose
        self.__model.clear()
        del train_x, train_y, test_x, test_y

    def trace_files(self) -> list:
        return self.__trace_filename


class PSGDPSExecutor(IExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        # wait
        self.__log = Logger('ParaServer'.format(node_id), log_to_file=True)
        self.__essential : IServerModel = None
        self.__setting : Settings = None
        self.__done : bool = False

    def requests(self) -> list:
        return [Req.Weights_And_Layers, Req.GlobalSettings]

    def satisfy(self, reply:list) -> list:
        # check list
        for obj in reply:

            if isinstance(obj, Settings):
                self.__add_setting(obj)

            elif isinstance(obj, IServerModel):
                self.__add_info(obj)

        return []

    def __add_setting(self, obj:Settings):
        self.__setting = obj
        # register global settings
        GlobalSettings.deprecated_default_settings = obj

    def __add_info(self, obj:IServerModel):
        self.__essential = obj

    def ready(self) -> bool:
        return self.__essential is not None \
                and self.__setting is not None

    def run(self, com:ICommunication_Controller) -> None:
        # build weights updater
        sgd_type = self.__essential.psgd_server_type
        iterator = range(len(self.__essential.codec_ctrl))
        codec_type = self.__essential.psgd_server_codec
        weights_updater = [
            {w:sgd_type(self.node_id, i, codec_type) for w in self.__essential.weights_types} for i in iterator
        ]
        # build transfer thread
        transfer = NTransfer(weights_updater, com, 0, self.__log)
        self.__log.log_message('Transfer thread is ready.')

        transfer.start_transfer()

        from utils.constants import Initialization_Server
        while set(com.available_clients) - {Initialization_Server} != set():
            sleep(7)

    def trace_files(self) -> list:
        return [self.__log.File_Name]

    def done(self) -> bool:
        return self.__done
