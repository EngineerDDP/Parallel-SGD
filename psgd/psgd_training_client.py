from threading import Thread
from time import sleep

import pandas as pd

from nn.metrics import CategoricalAccuracy
from nn.model import SequentialModel_v2
from nn.optimizer import ParallelSGDOptimizer
from profiles.settings import GlobalSettings
from psgd.transfer import NTransfer
from utils.log import Logger

from network.interfaces import ICommunication_Controller


class PSGDTraining_Parameter_Server(Thread):

    def __init__(self, model_init, com: ICommunication_Controller, ps_codec, ps_sgd_type, w_types, logger: Logger):

        super().__init__(name='Simulated parameter server process.')

        iNodeId = int(com.Node_Id)
        self.Com = com
        self.Logger = logger

        updater = [{} for _ in model_init]

        for layer_id in range(len(model_init)):
            for type in w_types:
                updater[layer_id][type] = ps_sgd_type(iNodeId, layer_id, ps_codec)

        self.Transfer = NTransfer(updater, self.Com)
        self.Logger.log_message('Init parameter server with codec {}, sync type {}.'.format(ps_codec, ps_sgd_type))

    def run(self):
        self.Transfer.start_transfer()
        while len(self.Com.available_clients()) != 1:
            sleep(1)


class PSGDTraining_Client(Thread):

    def __init__(self,
                 model_init,
                 loss,
                 codec_type,
                 sync_class,
                 com: ICommunication_Controller,
                 w_types,
                 tags,
                 train_x, train_y,
                 eval_x, eval_y,
                 batch_size,
                 epochs,
                 logger: Logger,
                 learn_rate=0.01,
                 target_acc=None):

        Thread.__init__(self, name='Simulated training process. Node: {}'.format(tags[0].Node_No))
        iNodeId = int(com.Node_Id)
        # pickle
        nn = model_init
        updater = [{} for i in model_init]

        for layer_id in range(len(nn)):
            for type in w_types:
                updater[layer_id][type] = sync_class(iNodeId, layer_id, codec_type)

        self.Batch_Size = batch_size
        self.Epochs = epochs

        self.Transfer = NTransfer(updater, com, logger)
        self.Optimizer = ParallelSGDOptimizer(tags=tags, batch_size=self.Batch_Size, com=self.Transfer, learn_rate=learn_rate)
        self.Model = SequentialModel_v2(nn, logger=logger)
        self.Model.compile(optimizer=self.Optimizer, loss=loss(), metrics=[CategoricalAccuracy()])

        self.Trace_Name = 'Trace_Node={}_Codec={}_R={}'.format(GlobalSettings.get_default().node_count, codec_type.__name__, GlobalSettings.get_default().redundancy)
        self.Log = logger
        self.Log_Header = logger.Title
        self.Log.log_message(self.Model.summary())

        self.Train_X = train_x
        self.Train_Y = train_y
        self.Eval_X = eval_x
        self.Eval_Y = eval_y

        self.Target_acc = target_acc
        self.Trace_Train = "training-{}.csv".format(self.Trace_Name)
        self.Trace_Eval = "eval-{}.csv".format(self.Trace_Name)

    def run(self):

        evaluation_history = []
        self.Transfer.start_transfer()
        # do until reach the target accuracy
        for i in range(self.Epochs):
            # change title
            self.Log.Title = self.Log_Header + "-Cyc{}".format(i)
            self.Model.fit(self.Train_X, self.Train_Y, epochs=1, batch_size=self.Batch_Size)
            # do tests
            evaluation = self.Model.evaluate(self.Eval_X, self.Eval_Y)
            if self.Target_acc is not None:
                # only one metric in model metrics list.
                # evaluation[0] refers to loss
                # evaluation[1] refers to accuracy.
                if evaluation[1] > self.Target_acc:
                    break
            evaluation_history.append(evaluation)

        training_history = self.Model.History
        # save training history data
        training_trace = pd.DataFrame(training_history, columns=self.Model.History_Title)
        training_trace.to_csv(self.Trace_Train, index=None)
        # save evaluation history data
        evaluation_trace = pd.DataFrame(evaluation_history, columns=self.Model.Evaluation_Title)
        evaluation_trace.to_csv(self.Trace_Eval, index=None)
        self.Log.log_message('Trace file has been saved to {}'.format(self.Trace_Name))

    def evaluate(self):
        result = self.Model.evaluate(self.Eval_X, self.Eval_Y)
        self.Log.log_message('Evaluate result: {}'.format(dict(zip(self.Model.History_Title[-len(result):], result))))
