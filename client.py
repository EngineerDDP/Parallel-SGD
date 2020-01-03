import time
import pandas as pd

from threading import Thread

from settings import GlobalSettings

from network.communications import CommunicationController
from network.agreements import General, Initialize, DefaultNodes, Data

from neuralnetworks.metrics import CategoricalAccuracy
from neuralnetworks.optimizer import ParallelSGDOptimizer, DelayedPSGDOptimizer
from neuralnetworks.model import SequentialModel_v2

from psgd.transfer import NTransfer

from codec.tag import Tag

from dataset.mnist_input import load_mnist

from log import Logger

import sys


class MNISTTrainingThread(Thread):

    def __init__(self, model_init, loss, codec_type, sync_class, com, w_types, tags, train_x, train_y, eval_x, eval_y, batch_size, epochs, logger, learn_rate=0.01, target_acc=None):

        Thread.__init__(self, name='Simulated training process. Node: {}'.format(tags[0].Node_No))

        # pickle
        nn = model_init
        updater = [{} for i in model_init]

        for layer_id in range(len(nn)):
            for type in w_types:
                updater[layer_id][type] = sync_class(com.Node_ID, layer_id, codec_type)

        self.Batch_Size = batch_size
        self.Epochs = epochs

        self.Transfer = NTransfer(updater, com, logger)
        self.Optimizer = ParallelSGDOptimizer(tags=tags, batch_size=self.Batch_Size, com=self.Transfer, learn_rate=learn_rate)
        self.Model = SequentialModel_v2(nn, logger=logger)
        self.Model.compile(optimizer=self.Optimizer, loss=loss(), metrics=[CategoricalAccuracy()])

        self.Trace_Name = 'Trace_Node={}_Codec={}_R={}'.format(GlobalSettings.get_default().NodeCount, codec_type.__name__, GlobalSettings.get_default().Redundancy)
        self.Log = logger

        self.Train_X = train_x
        self.Train_Y = train_y
        self.Eval_X = eval_x
        self.Eval_Y = eval_y

        self.Target_acc = target_acc

    def run(self):

        self.Transfer.start_transfer()
        # do until reach the target accuracy
        if self.Target_acc is not None:
            for i in range(len(self.Epochs)):
                self.Model.fit(self.Train_X, self.Train_Y, epochs=1, batch_size=self.Batch_Size)
                if self.Model.evaluate(self.Eval_X, self.Eval_Y)[1] > self.Target_acc:
                    break
        # just do
        else:
            self.Model.fit(self.Train_X, self.Train_Y, epochs=self.Epochs, batch_size=self.Batch_Size)

        history = self.Model.History
        trace = pd.DataFrame(history, columns=self.Model.History_Title)
        trace.to_csv("./training/{}.csv".format(self.Trace_Name), index=None)
        self.Log.log_message('Trace file has been saved to {}'.format(self.Trace_Name))

    def evaluate(self):
        result = self.Model.evaluate(self.Eval_X, self.Eval_Y)
        self.Log.log_message('Evaluate result: {}'.format(dict(zip(self.Model.History_Title, result))))

def main():

    if len(sys.argv) < 5:
        print('usage: client.py')
        print('\t --ip <ip address of initialization server>')
        print('\t --port <port of initialization server>')
        exit(-1)

    ip_addr = sys.argv[2]
    port = int(sys.argv[4])

    # Set remote
    CommunicationController.static_server_address = ip_addr
    CommunicationController.static_server_port = port
    # Communication controller
    con = CommunicationController()
    # Logger
    print('Establishing connection with {}:{}...'.format(ip_addr, port))
    con.establish_communication()
    print('Connection established.')
    # Logger
    log = Logger('Node {}'.format(con.Node_ID), log_to_file=False)
    log.log_message('Test Log...')

    # # take few
    # take = 60000
    # train_x = train_x[:take]
    # train_y = train_y[:take]
    log.log_message('Data loaded.')

    log.log_message('Initialing static environment...')
    con.send_one([DefaultNodes.Initialization_Server], {General.Type: Initialize.Init_Weight})

    sender, model_init_dic = con.get_one()

    log.log_message('Setup global settings.')
    GlobalSettings.set_default(
        model_init_dic[Initialize.Nodes],
        model_init_dic[Initialize.Redundancy],
        model_init_dic[Initialize.Batch_Size])

    log.log_message('Nodes: {}, Redundancy: {}, Batch size: {}.'.format(
        GlobalSettings.get_default().Nodes,
        GlobalSettings.get_default().Redundancy,
        GlobalSettings.get_default().Batch.Batch_Size))

    codec = model_init_dic[Initialize.CodeType]
    psgd = model_init_dic[Initialize.SyncClass]
    log.log_message('Codec: {}'.format(codec))
    log.log_message('Parallel Stochastic Gradient Descent: {}'.format(psgd))

    # log.log_message('Requiring data...')
    # con.send_one([DefaultNodes.Initialization_Server], {General.Type: Data.Type})
    # sender, data_init_dic = con.get_one()
    #
    log.log_message('Loading data...')
    # eval_x, eval_y = data_init_dic[Data.Eval_Data]
    # train_x, train_y = data_init_dic[Data.Train_Data]
    eval_x, eval_y = load_mnist(kind='t10k')
    train_x, train_y = load_mnist(kind='train')

    log.log_message('Initialing local runtime environment...')
    w_types = ['w', 'b']

    training_iterations = model_init_dic[Initialize.Epoches]
    loss_function_type = model_init_dic[Initialize.LOSS]
    learning_rate = model_init_dic[Initialize.Learn_Rate]
    target_accuracy = model_init_dic[Initialize.Target_Accuracy]

    log.log_message('Setup communication thread...')
    # helper = TransferHelper()
    # linker = HelperLink(transfer, helper)

    # tags to be trained
    tags = []

    for block in GlobalSettings.get_default().BlockAssignment.Node2Block[con.Node_ID]:

        tags.append(Tag(GlobalSettings.get_default().Batch, block, con.Node_ID,
                        set(GlobalSettings.get_default().BlockAssignment.Block2Node[block])))

    train = MNISTTrainingThread(model_init_dic[Initialize.Weight_Content],
                                loss_function_type, codec, psgd, con, w_types,
                                tags, train_x, train_y, eval_x, eval_y,
                                GlobalSettings.get_default().Batch.Batch_Size,
                                training_iterations, learn_rate=learning_rate, logger=log, target_acc=target_accuracy)

    log.log_message('Synchronizing timeline with cluster...')
    ready = False

    while not ready:
        con.send_one([-1], {General.Type: Initialize.Current_State})
        _, dic = con.get_one()
        ready = (dic[Initialize.Current_State] == Initialize.State_OK)
        time.sleep(1)

    log.log_message('State ok, starting all process...')

    begin = time.time()

    # linker.start()
    train.start()
    train.join()

    # Record execution time
    end = time.time()    # Close connection with server
    con.close()

    print('-----------------------------------------------------------')
    log.log_message('Execution complete, time:{}'.format(end - begin))
    print('-----------------------------------------------------------')

    train.evaluate()

    print('-------------------------Done------------------------------')


if __name__ == '__main__':
    """
        Usage:
        python client.py --ip 121.248.202.131 --port 15387
    """

    main()