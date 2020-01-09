from codec.interfaces import yield_none
from codec.interfaces import ICommunicationCtrl
from codec.pacodec import PAServerCompack, PAClientComPack
from log import Logger

from profiles.settings import GlobalSettings


class GradDiffParaServerCodec(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('PAS')):

        ICommunicationCtrl.__init__(self)
        self.Node_ID = node_id
        # save PA current state
        self.Current_Weights = 0
        self.Log = logger
        # save previous state for each node
        self.Bak_Weights_Node = {}
        # init w_bak
        for key in GlobalSettings.get_default().nodes:
            self.Bak_Weights_Node[key] = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        return yield_none()

    def receive_blocks(self, json_dict):
        """
            PA Server receive a json_dict and send back a request
        :param json_dict:
        :return:
        """
        # analyze received data
        compack = PAClientComPack.from_dictionary(json_dict)
        # get last state of working node
        last_state = self.Bak_Weights_Node[compack.Node_ID]
        # update global current state
        self.Current_Weights = self.Current_Weights + compack.Content
        # get difference
        grad_diff = self.Current_Weights - last_state
        # update last state of working node
        self.Bak_Weights_Node[compack.Node_ID] = self.Current_Weights
        # build communication package
        comback = PAServerCompack.compose_compack(grad_diff)

        yield ([compack.Node_ID], comback.to_dictionary())


class ParaServerCodec(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('PAS')):

        ICommunicationCtrl.__init__(self)
        self.Node_ID = node_id
        # save PA current state
        self.Current_Weights = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        return yield_none()

    def receive_blocks(self, json_dict):
        """
            PA Server receive a json_dict and send back a request
        :param json_dict:
        :return:
        """
        # analyze received data
        compack = PAClientComPack.from_dictionary(json_dict)
        # update global current state
        self.Current_Weights = self.Current_Weights - compack.Content
        comback = PAServerCompack.compose_compack(self.Current_Weights.astype('double'))

        yield ([compack.Node_ID], comback.to_dictionary())
