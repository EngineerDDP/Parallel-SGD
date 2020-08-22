from threading import Thread
from utils.log import Logger

from psgd.interfaces import ITransfer, ReadTimeOut
from network.interfaces import ICommunication_Controller


class NTransfer(ITransfer):
    """
        string identifier used in this class.
        identifier used for json key tag.
        wrote and solved only within this code scope.
    """
    STR_LAYER_NO = 'NLayer_NO'
    STR_W_TYPE = 'NW_Type'
    INT_RETRY_LIMIT = 5

    def __init__(self, weights_ctrl, com: ICommunication_Controller, logger=Logger('Default Transfer')):
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

        self.working_thread = Thread(name='Transfer thread for node {}.' \
                                     .format(self.communication_process.Node_Id), target=self.__run)
        self.Node_ID = com.Node_Id
        self.__log = logger

    def put_weights(self, content, tag, w_type='w'):
        """
            Put weights instantly
            No waiting
        """
        # Copy tag
        update_packs = self.type_weights_controller[tag.Layer_No][w_type].update_weights(content, tag)
        for update_pack in update_packs:
            sender, dic = update_pack
            self.__send(sender, dic, tag.Layer_No, w_type)

    def get_weights(self, tag, w_type='w'):
        """
            Acquire weights instantly
            No waiting
        """
        try:
            return self.type_weights_controller[tag.Layer_No][w_type].require_weights(tag)
        except ReadTimeOut as e:
            if e.retry() is None:
                self.__log.log_error('Time out while getting result, retry not available.')
                raise TimeoutError('Time out while get weights.')
            for sender, dic in e.retry():
                self.__send(sender, dic, tag.Layer_No, w_type)
                self.__log.log_error('Message retry to node {}'.format(sender))
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
                # blocking other format
                if not isinstance(dic, dict):
                    continue
                # quit processing if the object is not sent by the class instance like NTransfer
                try:
                    layer_no = dic[NTransfer.STR_LAYER_NO]
                    w_type = dic[NTransfer.STR_W_TYPE]
                    update_packs = self.type_weights_controller[layer_no][w_type].accept_data(dic)
                    # self.Log.log_message('Message accepted.')
                    if update_packs is None:
                        continue
                    for update_pack in update_packs:
                        sender, dic = update_pack
                        self.__send(sender, dic, layer_no, w_type)
                        # self.Log.log_message('Message back to node {}'.format(sender))
                except KeyError as e:
                    # print DEBUG message
                    import sys
                    import traceback
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    exc_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
                    for line in exc_tb:
                        self.__log.log_message(line)
                    # print DEBUG message
        except OSError as e:
            self.__log.log_message('Transfer thread report an error: {}'.format(e))
        except ValueError:
            pass
        finally:
            self.__log.log_message('Transfer thread exited safely.')
