from threading import Thread

import pandas as pd

from nn.metrics import CategoricalAccuracy
from nn.model import SequentialModel_v2
from nn.optimizer import ParallelSGDOptimizer
from profiles.settings import GlobalSettings
from psgd.transfer import NTransfer


class PSGDTraining_Client(Thread):

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
        self.Model.summary()

        self.Trace_Name = 'Trace_Node={}_Codec={}_R={}'.format(GlobalSettings.get_default().node_count, codec_type.__name__, GlobalSettings.get_default().redundancy)
        self.Log = logger
        self.Log_Header = logger.Title

        self.Train_X = train_x
        self.Train_Y = train_y
        self.Eval_X = eval_x
        self.Eval_Y = eval_y

        self.Target_acc = target_acc
        self.Trace_Train = "training-{}.csv".format(self.Trace_Name)
        self.Trace_Eval = "./eval-{}.csv".format(self.Trace_Name)

    def run(self):

        evaluation_history = []
        self.Transfer.start_transfer()
        # do until reach the target accuracy
        for i in range(self.Epochs):
            self.Model.fit(self.Train_X, self.Train_Y, epochs=1, batch_size=self.Batch_Size)
            # change title
            self.Log.Title = self.Log_Header + "-Cyc{}".format(i)
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


# if __name__ == '__main__':
#     """
#         Usage:
#         python psgd_training_client.py --ip 121.248.202.131 --port 15387
#     """
#
#     if len(sys.argv) < 5:
#         print('usage: psgd_training_client.py')
#         print('\t --ip <ip address of initialization server>')
#         print('\t --port <port of initialization server>')
#         sys.exit(-1)
#
#     ip_addr = sys.argv[2]
#     port = int(sys.argv[4])
#
#     # Set remote
#     CommunicationController.static_server_address = ip_addr
#     CommunicationController.static_server_port = port
#     # Communication controller
#     con = CommunicationController()
#     # Logger
#     print('Establishing connection with {}:{}...'.format(ip_addr, port))
#     con.establish_communication()
#     print('Connection established.')
#     # Logger
#     log = Logger('Node {}'.format(con.Node_ID), log_to_file=False)
#     log.log_message('Test Log...')
#
#     # # take few
#     # take = 60000
#     # train_x = train_x[:take]
#     # train_y = train_y[:take]
#     log.log_message('Data loaded.')
#
#     log.log_message('Initialing static environment...')
#     con.send_one([DefaultNodes.Initialization_Server], {General.Type: Initialize.Init_Weight})
#
#     sender, model_init_dic = con.get_one()
#
#     log.log_message('Setup global settings.')
#     GlobalSettings.set_default(
#         model_init_dic[Initialize.Nodes],
#         model_init_dic[Initialize.Redundancy],
#         model_init_dic[Initialize.Batch_Size],
#         model_init_dic[Initialize.Block_Assignment])
#
#     log.log_message('Nodes: {}, Redundancy: {}, Batch size: {}, Assignments: {}.'.format(
#         GlobalSettings.get_default().nodes,
#         GlobalSettings.get_default().redundancy,
#         GlobalSettings.get_default().batch.batch_size,
#         model_init_dic[Initialize.Block_Assignment].__name__))
#
#     codec = model_init_dic[Initialize.CodeType]
#     psgd = model_init_dic[Initialize.SyncClass]
#     log.log_message('Codec: {}'.format(codec))
#     log.log_message('Parallel Stochastic Gradient Descent: {}'.format(psgd))
#
#     # log.log_message('Requiring data...')
#     # con.send_one([DefaultNodes.Initialization_Server], {General.Type: Data.Type})
#     # sender, data_init_dic = con.get_one()
#     #
#     log.log_message('Loading data...')
#     # eval_x, eval_y = data_init_dic[Data.Eval_Data]
#     # train_x, train_y = data_init_dic[Data.Train_Data]
#     eval_x, eval_y = load_mnist(kind='t10k')
#     train_x, train_y = load_mnist(kind='train')
#
#     log.log_message('Initialing local runtime environment...')
#     w_types = ['w', 'b']
#
#     training_iterations = model_init_dic[Initialize.Epoches]
#     loss_function_type = model_init_dic[Initialize.LOSS]
#     learning_rate = model_init_dic[Initialize.Learn_Rate]
#     target_accuracy = model_init_dic[Initialize.Target_Accuracy]
#
#     log.log_message('Setup communication thread...')
#     # helper = TransferHelper()
#     # linker = HelperLink(transfer, helper)
#
#     # tags to be trained
#     tags = []
#
#     for block in GlobalSettings.get_default().block_assignment.node_2_block[con.Node_ID]:
#
#         tags.append(Tag(GlobalSettings.get_default().batch, block, con.Node_ID,
#                         set(GlobalSettings.get_default().block_assignment.block_2_node[block])))
#
#     train = MNISTTrainingThread(model_init_dic[Initialize.Weight_Content],
#                                 loss_function_type, codec, psgd, con, w_types,
#                                 tags, train_x, train_y, eval_x, eval_y,
#                                 GlobalSettings.get_default().batch.batch_size,
#                                 training_iterations, learn_rate=learning_rate, logger=log,
#                                 target_acc=target_accuracy)
#
#     log.log_message('Synchronizing timeline with cluster...')
#     ready = False
#
#     while not ready:
#         con.send_one([-1], {General.Type: Initialize.Current_State})
#         _, dic = con.get_one()
#         ready = (dic[Initialize.Current_State] == Initialize.State_OK)
#         time.sleep(1)
#
#     log.log_message('State ok, starting all process...')
#
#     begin = time.time()
#
#     # linker.start()
#     train.start()
#     train.join()
#
#     # Record execution time
#     end = time.time()    # Close connection with server
#     con.close()
#
#     print('-----------------------------------------------------------')
#     log.log_message('Execution complete, time:{}'.format(end - begin))
#     print('-----------------------------------------------------------')
#
#     train.evaluate()
#
#     print('-------------------------Done------------------------------')
