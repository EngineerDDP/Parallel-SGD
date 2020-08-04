from codec.interfaces import ICommunicationCtrl, IComPack, yield_none, NetEncapsulation

from utils.constants import Parameter_Server
from utils.log import Logger

from profiles.settings import GlobalSettings

class PAClientCodec(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('None')):

        ICommunicationCtrl.__init__(self)

        self.Node_id = node_id
        self.Log = logger

    def dispose(self):
        """
            Nothing to do.
        """
        pass

    def update_blocks(self, block_weight):

        send, pack = PAClientComPack.compose_compack(block_weight, self.Node_id)
        pack = PAClientComPack.to_dictionary(pack)
        yield NetEncapsulation(send, pack)

    def receive_blocks(self, json_dict):

        compack = PAServerCompack.from_dictionary(json_dict)
        self.set_result(compack.Content)

        return yield_none()


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

        yield NetEncapsulation(compack.Node_ID, comback.to_dictionary())


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
        self.Current_Weights = self.Current_Weights + compack.Content
        comback = PAServerCompack.compose_compack(self.Current_Weights.astype('double'))

        yield NetEncapsulation(compack.Node_ID, comback.to_dictionary())


class PAClientComPack(IComPack):

    def __init__(self, node, layer, block, content):
        self.Node_ID = node
        self.Layer_ID = layer
        self.Block_ID = block
        self.Content = content

    def to_dictionary(self):
        dic = {
            'Node_ID':  self.Node_ID,
            'Layer_ID': self.Layer_ID,
            'Block_ID': self.Block_ID,
            'Content':  self.Content
        }
        return dic

    @staticmethod
    def from_dictionary(cls):
        node_id = cls['Node_ID']
        layer_id = cls['Layer_ID']
        block_id = cls['Block_ID']
        content = cls['Content']
        return PAClientComPack(node_id, layer_id, block_id, content)

    @staticmethod
    def compose_compack(block_weights, node_id=None):

        send_target = [Parameter_Server]
        pack = PAClientComPack(node_id, block_weights.Layer_ID, block_weights.Block_ID, block_weights.Content)

        return send_target, pack

    @staticmethod
    def decompose_compack(self, params=None):
        pass


class PAServerCompack(PAClientComPack):

    def __init__(self, node, layer, block, content):

        PAClientComPack.__init__(self, node, layer, block, content)

    @staticmethod
    def compose_compack(content, params=None):
        return PAServerCompack(-1, -1, -1, content)

