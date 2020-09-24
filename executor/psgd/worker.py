import time
from typing import List

import pandas as pd

from codec import GlobalSettings
from dataset.interfaces import IDataset
from executor.abstract import AbsExecutor
from executor.psgd.net_package import IPSGDOptimize, misc_package, Requests, Req
from executor.psgd.net_package import net_model, net_setting
from models import IRequestPackage
from network import ICommunication_Controller
from nn import IModel
from nn.data import PSGDBlockDataFeeder
from profiles import ISetting
from profiles.interface import IBatchIter
from psgd.interface import ITransfer
from utils.log import Logger


class PSGDWorkerExecutor(AbsExecutor):

    def __init__(self, node_id, offset):
        super().__init__(node_id, offset)
        self.__log = Logger('Fit-{}'.format(node_id), log_to_file=True)
        self.__trace_filename = [self.__log.File_Name]
        # waiting for those
        self.__model: [IModel] = None
        self.__optimizer: [IPSGDOptimize] = None
        self.__setting: [ISetting] = None
        self.__batch_iter: [IBatchIter] = None
        self.__trans: [ITransfer] = None
        self.__data: [IDataset] = None
        self.__misc: [misc_package] = None
        self.__done: bool = False

    def requests(self) -> List[IRequestPackage]:
        return [Requests(Req.Setting), Requests(Req.Model), Requests(Req.Optimizer),
                Requests(Req.Transfer), Requests(Req.Data_Package), Requests(Req.Other_Stuff)]

    def satisfy(self, reply:list) -> list:
        unsatisfied = []
        # check list
        for obj in reply:

            if isinstance(obj, net_setting):
                GlobalSettings.deprecated_default_settings = obj.setting()

            if isinstance(obj, net_model):
                self.__model = obj.model
                self.__batch_iter = obj.batch_iter

            if isinstance(obj, IPSGDOptimize):
                self.__optimizer = obj

            if isinstance(obj, ITransfer):
                self.__trans = obj

            if isinstance(obj, misc_package):
                self.__misc = obj

            if isinstance(obj, IDataset):
                if not obj.check():
                    unsatisfied.append(Requests(Req.Data_Content))
                else:
                    self.__data = obj

        return unsatisfied

    def ready(self) -> bool:
        return self.__model and self.__optimizer and self.__trans and self.__misc \
                and self.__data and GlobalSettings.deprecated_default_settings

    def done(self) -> bool:
        return self.__done

    def start(self, com:ICommunication_Controller) -> None:
        # type assertions.
        assert isinstance(self.__optimizer, IPSGDOptimize) and \
               isinstance(self.__model, IModel) and \
               isinstance(self.__data, IDataset) and \
               isinstance(self.__setting, ISetting) and \
               isinstance(self.__trans, ITransfer) and \
               isinstance(self.__batch_iter, IBatchIter)
        # get dataset
        train_x, train_y, test_x, test_y = self.__data.load()
        self.__log.log_message('Dataset is ready, type: ({})'.format(self.__data))
        # build data feeder
        feeder = PSGDBlockDataFeeder(train_x, train_y, batch_iter=self.__batch_iter, block_ids=self.__setting.blocks)
        # assemble optimizer
        self.__optimizer.assemble(transfer=self.__trans, block_mgr=feeder)
        # compile model
        self.__model.compile(self.__optimizer)
        # summary
        summary = self.__model.summary()
        self.__log.log_message(summary)
        trace_head = '{}-N({})'.format(self.__misc.mission_title, self.node_id)
        self.__log.log_message('Model set to ready.')

        log_head = self.__log.Title
        # start !
        GlobalSettings.deprecated_global_logger = self.__log
        self.__trans.start_transfer(com, group_offset=self.group[0], printer=self.__log)
        # record data
        time_start = time.time()
        data_send_start = com.Com.bytes_sent
        data_recv_start = com.Com.bytes_read

        evaluation_history = []
        title = []
        # do until reach the target accuracy
        for i in range(self.__misc.epoch):
            # change title
            self.__log.Title = log_head + "-Epo-{}".format(i + 1)
            history = self.__model.fit(feeder, epoch=1, printer=self.__log)
            # do tests
            r = self.__model.evaluate(test_x, test_y)
            title = r.keys()
            row = r.values()
            self.__log.log_message('Evaluate result: {}'.format(r))
            evaluation_history.append(row)

            if self.__misc.target_acc is not None:
                # only one metric in model metrics list.
                # evaluation[0] refers to loss
                # evaluation[1] refers to accuracy.
                if r[1] > self.__misc.target_acc:
                    break

        # record data
        time_end = time.time()
        data_sent_end = com.Com.bytes_sent
        data_recv_end = com.Com.bytes_read

        training_history = self.__model.fit_history()
        # save training history data
        training_name = "TR-" + trace_head + ".csv"
        training_trace = pd.DataFrame(training_history.history, columns=training_history.title)
        training_trace.to_csv(training_name, index=False)
        # save evaluation history data
        evaluation_name = "EV-" + trace_head + ".csv"
        evaluation_trace = pd.DataFrame(evaluation_history, columns=title)
        evaluation_trace.to_csv(evaluation_name, index=False)
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

