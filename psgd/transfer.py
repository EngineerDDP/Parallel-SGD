from threading import Thread
from log import Logger

from network.agreements import General
from network.agreements import Transfer as TransferAgreements

from psgd.interfaces import IDispatcher, IParallelSGD, ITransfer


class NTransfer(ITransfer):

    """
        string identifier used in this class.
        identifier used for json key tag.
        wrote and solved only within this code scope.
    """
    STR_LAYER_NO = 'NLayer_NO'
    STR_W_TYPE = 'NW_Type'

    def __init__(self, weights_ctrl, com, logger=Logger('Default Transfer')):
        """
            build a transfer controller for transferring data between local ML process and
            remote server process.
            Each training process has exactly one transfer instance.
        :param weights_ctrl: list of controller for transferring data, identified for each layer, each type.
        :param com: Network Communication Controller
        """

        # formatted as weights_initial
        self.type_weights_controller = weights_ctrl
        self.communication_process = com

        self.working_thread = Thread(name='Transfer thread for node {}.'\
                                     .format(self.communication_process.Node_ID), target=self.__run)
        self.Node_ID = com.Node_ID
        self.Log = logger

    def put_weights(self, content, tag, w_type='w'):
        """
            Put weights instantly
            No waiting
        """
        # Copy tag
        update_pack = self.type_weights_controller[tag.Layer_No][w_type].update_weights(content, tag)
        if update_pack is not None:
            sender, dic = update_pack
            self.__send(sender, dic, tag.Layer_No, w_type)
            self.Log.log_message('Message to node {}'.format(sender))

    def get_weights(self, tag, w_type='w'):
        """
            Acquire weights instantly
            No waiting
        """
        return self.type_weights_controller[tag.Layer_No][w_type].require_weights(tag)

    def start_transfer(self):
        """
            Start transferring data between psgd controller and communication process.
            reference call (IParallelSGD.accept_data()) without sync check, is not thread safe call.
        :return: None
        """
        self.working_thread.start()

    def __send(self, target, dic, layer_no, w_type):
        """
            Write tag and send
        """
        # skip none
        if len(target) == 0:
            return
        # write tag
        dic[NTransfer.STR_LAYER_NO] = layer_no
        dic[NTransfer.STR_W_TYPE] = w_type
        dic[General.Type] = TransferAgreements.Type
        self.communication_process.send_one(target, dic)

    def __run(self):
        """
            Working thread.
            Quit if self.communication_process is not alive.
        :return: None
        """
        try:
            while not self.communication_process.is_closed():
                sender, dic = self.communication_process.get_one()
                self.Log.log_message('Recv from node {}'.format(dic[General.From]))
                # quit processing if the object is not sent by the class instance like NTransfer
                try:
                    layer_no = dic[NTransfer.STR_LAYER_NO]
                    w_type = dic[NTransfer.STR_W_TYPE]
                    update_pack = self.type_weights_controller[layer_no][w_type].accept_data(dic)
                    self.Log.log_message('Message accepted.')
                    if update_pack is not None:
                        sender, dic = update_pack
                        self.__send(sender, dic, layer_no, w_type)
                        self.Log.log_message('Message back to node {}'.format(sender))
                except KeyError:
                    continue
        except TypeError:
            pass
        print('Transfer thread exited safely.')
