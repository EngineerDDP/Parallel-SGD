import time

from threading import Thread

from settings import GlobalSettings

from network.communications import CommunicationController
from network.agreements import General, Initialize, DefaultNodes, Data

from neuralnetworks.layers import FCLayer
from neuralnetworks.optimizer import ParallelGradientDecentOptimizer, DelayedParallelGradientDecentOptimizer
from neuralnetworks.model import Trace_Model, Normal_Model

from psgd.transfer import NTransfer
from psgd.interfaces import IParallelSGD

from codec.tag import Tag

from dataset.mnist_input import load_mnist

from log import Logger

import sys


class MNISTTrainingThread(Thread):

    def __init__(self, model_init, losses, codec_type, sync_class, com, w_types, tags, train_x, train_y, eval_x, eval_y, batch_size, epoches, learnrate=0.01, target_acc=0.96):

        Thread.__init__(self, name='Simulated training process. Node: {}'.format(tags[0].Node_No))

        # pickle
        nn = model_init
        updater = [{} for i in model_init]

        for layer_id in range(len(nn)):
            for type in w_types:
                updater[layer_id][type] = sync_class(com.Node_ID, layer_id, codec_type)

        self.Batch_Size = batch_size
        self.Epoches = epoches

        self.Transfer = NTransfer(updater, com)
        self.Optimizer = ParallelGradientDecentOptimizer(losses(), nn, tags, self.Transfer, learnrate)
        if tags[0].Node_No == 0:
            self.Model = Trace_Model(nn, self.Optimizer, target_acc=target_acc, trace_name='Trace_Node={}_Codec={}-'.format(GlobalSettings.getDefault().NodeCount, codec_type.__name__))
        else:
            self.Model = Normal_Model(nn, self.Optimizer)

        self.Train_X = train_x
        self.Train_Y = train_y
        self.Eval_X = eval_x
        self.Eval_Y = eval_y

    def run(self):

        self.Transfer.start_transfer()
        self.Model.fit(self.Train_X, self.Train_Y, self.Epoches, self.Batch_Size, val_x=self.Eval_X, val_y=self.Eval_Y)


def main():

    # ----------------------------!!!!!!!!!!!!----------------------------
    # GlobalSettings.setDefault(4, 1, 96)
    # ----------------------------!!!!!!!!!!!!----------------------------

    # Set remote
    CommunicationController.static_server_address = '121.248.202.131'
    CommunicationController.static_server_port = 15387
    # Communication controller
    con = CommunicationController()
    # Logger
    print('Establishing connection with {}...'.format(con.static_server_address))
    con.establish_communication()
    print('Connection established.')
    # Logger
    log = Logger('Node {}'.format(con.Node_ID), False)
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
    GlobalSettings.setDefault(
        model_init_dic[Initialize.Nodes],
        model_init_dic[Initialize.Redundancy],
        model_init_dic[Initialize.Batch_Size])

    log.log_message('Nodes: {}, Redundancy: {}, Batch size: {}.'.format(
        GlobalSettings.getDefault().Nodes,
        GlobalSettings.getDefault().Redundancy,
        GlobalSettings.getDefault().Batch.Batch_Size))

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

    EPOCHES = model_init_dic[Initialize.Epoches]
    Losses = model_init_dic[Initialize.LOSS]
    LR = model_init_dic[Initialize.Learn_Rate]

    log.log_message('Setup communication thread...')
    # helper = TransferHelper()
    # linker = HelperLink(transfer, helper)

    # tags to be trained
    tags = []

    for block in GlobalSettings.getDefault().BlockAssignment.Node2Block[con.Node_ID]:

        tags.append(Tag(GlobalSettings.getDefault().Batch, block, con.Node_ID,
                  set(GlobalSettings.getDefault().BlockAssignment.Block2Node[block])))

    train = MNISTTrainingThread(model_init_dic[Initialize.Weight_Content],
                                Losses, codec, psgd, con, w_types,
                                tags, train_x, train_y, eval_x, eval_y,
                                GlobalSettings.getDefault().Batch.Batch_Size,
                                EPOCHES, learnrate=LR)

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
    log.log_message('Time used in data transfer, time:{}'.format(train.Optimizer.total_non_execution_time))
    print('-----------------------------------------------------------')

    train.Model.evalute(eval_x, eval_y)

    print('-------------------------Done------------------------------')


if __name__ == '__main__':

    main()