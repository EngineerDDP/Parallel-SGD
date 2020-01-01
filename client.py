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

    def __init__(self, model_init, losses, codec_type, sync_class, com, w_types, tags, train_x, train_y, eval_x, eval_y, batch_size, epoches, logger, learnrate=0.01):

        Thread.__init__(self, name='Simulated training process. Node: {}'.format(tags[0].Node_No))

        # pickle
        nn = model_init
        updater = [{} for i in model_init]

        for layer_id in range(len(nn)):
            for type in w_types:
                updater[layer_id][type] = sync_class(com.Node_ID, layer_id, codec_type)

        self.Batch_Size = batch_size
        self.Epoches = epoches

        self.Transfer = NTransfer(updater, com, logger)
        self.Optimizer = ParallelGradientDecentOptimizer(losses(), nn, tags, self.Transfer, learnrate)
        self.Model = Trace_Model(nn, self.Optimizer, logger=logger, trace_name='Trace_Node={}_Codec={}_R={}'.format(GlobalSettings.get_default().NodeCount, codec_type.__name__, GlobalSettings.get_default().Redundancy))

        self.Train_X = train_x
        self.Train_Y = train_y
        self.Eval_X = eval_x
        self.Eval_Y = eval_y

    def run(self):

        self.Transfer.start_transfer()
        self.Model.fit(self.Train_X, self.Train_Y, self.Epoches, self.Batch_Size)


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

    EPOCHES = model_init_dic[Initialize.Epoches]
    Losses = model_init_dic[Initialize.LOSS]
    LR = model_init_dic[Initialize.Learn_Rate]

    log.log_message('Setup communication thread...')
    # helper = TransferHelper()
    # linker = HelperLink(transfer, helper)

    # tags to be trained
    tags = []

    for block in GlobalSettings.get_default().BlockAssignment.Node2Block[con.Node_ID]:

        tags.append(Tag(GlobalSettings.get_default().Batch, block, con.Node_ID,
                        set(GlobalSettings.get_default().BlockAssignment.Block2Node[block])))

    train = MNISTTrainingThread(model_init_dic[Initialize.Weight_Content],
                                Losses, codec, psgd, con, w_types,
                                tags, train_x, train_y, eval_x, eval_y,
                                GlobalSettings.get_default().Batch.Batch_Size,
                                EPOCHES, learnrate=LR, logger=log)

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
    """
        Usage:
        python client.py --ip 121.248.202.131 --port 15387
    """

    main()