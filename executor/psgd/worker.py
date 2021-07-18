import time

import pandas as pd

import codec
import executor.abstract as abstract
import executor.communication as communication
import nn
from executor.psgd.net_package import *
from nn.data import PSGDBlockDataFeeder
from nn.model import Model
from utils.log import Logger


class PSGDWorkerExecutor(abstract.AbsExecutor):

    def __init__(self, node_id: int, working_group: set, initializer_id: int = -1):
        super().__init__(node_id, working_group, initializer_id)
        self.__log = Logger('Fit-{}'.format(node_id), log_to_file=True)
        self.__trace_filename = [self.__log.File_Name]
        # waiting for those
        self.__model: [Model] = None
        self.__optimizer: [IPSGDOpContainer] = None
        self.__batch_iter: [IBatchIter] = None
        self.__trans: [ITransfer] = None
        self.__data: [IDataset] = None
        self.__misc: [misc_package] = None
        self.__done: bool = False

    def requests(self) -> List[object]:
        """
            先请求独立项，最后请求Transfer
        """
        return [Req.Setting,
                Req.Model,
                Req.Optimizer,
                Req.Data_Package,
                Req.Other_Stuff,
                Req.Extra_Content]

    def satisfy(self, reply: list) -> list:
        unsatisfied = []
        # check list
        for obj in reply:

            if isinstance(obj, net_setting):
                codec.GlobalSettings.deprecated_default_settings = obj.setting()

            if isinstance(obj, extra_package):
                codec.GlobalSettings.global_parameters = obj.acquire()  # Extra package 作为 transfer 的前置条件
                unsatisfied.append(Req.Transfer)

            if isinstance(obj, net_model):
                self.__model = obj.model
                self.__batch_iter = obj.batch_iter

            if isinstance(obj, IPSGDOpContainer):
                self.__optimizer = obj

            if isinstance(obj, ITransfer):
                self.__trans = obj

            if isinstance(obj, misc_package):
                self.__misc = obj

            if isinstance(obj, IDataset):
                if not obj.check():
                    unsatisfied.append(Req.Data_Content)
                else:
                    self.__data = obj

        return unsatisfied

    def ready(self) -> bool:
        return self.__check()[0]

    def __check(self) -> Tuple[bool, List[str]]:
        status = []
        s1 = isinstance(self.__optimizer, IPSGDOpContainer)
        status.append("Optimizer:{}".format("OK" if s1 else "ABSENT"))
        s2 = isinstance(self.__model, IModel)
        status.append("Model:{}".format("OK" if s2 else "ABSENT"))
        s3 = isinstance(self.__data, IDataset)
        status.append("Dataset:{}".format("OK" if s3 else "ABSENT"))
        s4 = isinstance(self.__misc, misc_package)
        status.append("Others:{}".format("OK" if s4 else "ABSENT"))
        s5 = isinstance(self.__trans, ITransfer)
        status.append("Transfer:{}".format("OK" if s5 else "ABSENT"))
        s6 = isinstance(self.__batch_iter, IBatchIter)
        status.append("Batch Iterator:{}".format("OK" if s6 else "ABSENT"))
        s7 = isinstance(codec.GlobalSettings.deprecated_default_settings, ISetting)
        status.append("Settings:{}".format("OK" if s7 else "ABSENT"))
        s8 = isinstance(codec.GlobalSettings.global_parameters, dict)
        status.append("Extra Parameters:{}".format("OK" if s8 else "ABSENT"))
        return s1 and s2 and s3 and s4 and s5 and s6 and s7 and s8, status

    def done(self) -> bool:
        return self.__done

    def start(self, com: communication.Communication) -> None:
        state, report = self.__check()
        self.__log.log_message("Ready:{} \n\t Check List:\n\t\t--> {}".format(state, "\n\t\t--> ".join(report)))
        # get dataset
        train_x, train_y, test_x, test_y = self.__data.load()
        self.__log.log_message('Dataset is ready, type: ({})'.format(self.__data))
        # build data feeder
        block_ids = codec.GlobalSettings.get_default().node_2_block[self.node_id]
        feeder = PSGDBlockDataFeeder(train_x, train_y, batch_iter=self.__batch_iter, block_ids=block_ids)
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
        codec.GlobalSettings.deprecated_global_logger = self.__log
        self.__trans.start_transfer(com, group_offset=list(self.group)[0], printer=self.__log, node_id=self.node_id)
        # record data
        time_start = time.time()
        data_send_start = com.bytes_sent
        data_recv_start = com.bytes_read

        evaluation_history = []
        title = []
        r = {}
        # this message will start the progress reporting
        com.report_progress(0)
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

            # report progress
            com.report_progress(int(100 * ((i + 1) / self.__misc.epoch)))

        # record data
        time_end = time.time()
        data_sent_end = com.bytes_sent
        data_recv_end = com.bytes_read

        training_history = self.__model.fit_history()
        # save training history data
        training_name = "TR-" + trace_head + ".csv"
        training_trace = pd.DataFrame(training_history.history, columns=training_history.title)
        training_trace.to_csv(training_name, index=False)
        # save evaluation history data
        evaluation_name = "EV-" + trace_head + ".csv"
        evaluation_trace = pd.DataFrame(evaluation_history, columns=title)
        evaluation_trace.to_csv(evaluation_name, index=False)
        # save model
        model_name = "MODEL-" + trace_head + ".model"
        self.__model.compile(nn.gradient_descent.SGDOptimizer(learn_rate=1e-5))
        self.__model.save(model_name)
        self.__trace_filename.append(training_name)
        self.__trace_filename.append(evaluation_name)
        self.__trace_filename.append(model_name)

        self.__log.log_message('Execution complete, time: {}.'.format(time_end - time_start))
        self.__log.log_message('Execution complete, Total bytes sent: {}.'.format(data_sent_end - data_send_start))
        self.__log.log_message('Execution complete, Total bytes read: {}.'.format(data_recv_end - data_recv_start))
        self.__log.log_message('Trace file has been saved to {}.'.format(trace_head))

        self.__log.flush()
        # set marker
        self.__done = True
        # dispose
        self.__model.clear()
        del train_x, train_y, test_x, test_y
        del self.__model, self.__log

        # return last evaluation result
        return r

    def trace_files(self) -> list:
        return self.__trace_filename
