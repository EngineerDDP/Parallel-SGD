from codec.essential import BatchWeight
from codec.essential import BlockWeight
from codec.interfaces import ICommunicationCtrl, IComPack

from network.agreements import DefaultNodes

from server_util.init_model import ServerUtil
from settings import GlobalSettings

from log import Logger

class PACodec(ICommunicationCtrl):

    def __init__(self, node_id, layer_id, logger=Logger('None')):

        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id
        self.Layer_ID = layer_id

        self.Log = logger

    def dispose(self):
        """
            Nothing to do.
        """
        pass

    def update_blocks(self, block_weight):

        send, pack = PAClientComPack.compose_compack(block_weight)

        return send, pack

    def receive_blocks(self, json_dict):

        compack = PAServerCompack.decompose_compack(json_dict)
        self.updated_weight_buffer = compack.Content


class PAClientComPack(IComPack):

    def __init__(self, node, layer, block, content):

        self.Node_ID = node
        self.Layer_ID = layer
        self.Block_ID = block
        self.Content = content

    def to_dictionary(cls):

        dic = {
            'Node_ID': cls.Node_ID,
            'Layer_ID': cls.Layer_ID,
            'Block_ID':cls.Block_ID,
            'Content':cls.Content
        }
        return dic

    def from_dictionary(cls):

        node_id = cls['Node_ID']
        layer_id = cls['Layer_ID']
        block_id = cls['Block_ID']
        content = cls['Content']

        return PAClientComPack(node_id, layer_id, block_id, content)

    def compose_compack(blockweights, params=None):

        send_target = DefaultNodes.Parameter_Server
        pack = PAClientComPack(blockweights.Node_ID, BlockWeight.Layer_ID, BlockWeight.Block_ID, BlockWeight.Content)

        return send_target, pack

    def decompose_compack(cls, params=None):

        return PAClientComPack.from_dictionary(cls)


class PAServerCodec(ICommunicationCtrl):

    def __init__(self, node_id, layer_id, logger=Logger('PAS')):

        super().__init__()

        self.Node_ID = node_id
        self.Layer_ID = layer_id
        self.Learn_Rate = ServerUtil.learn_rate()

        self.Bak_Weights_Node = {}

        self.Log = logger

        # init w_bak
        for key in GlobalSettings.getDefault().Nodes:
            self.Bak_Weights_Node[key] = 0

    def dispose(self):
        pass

    def update_blocks(self, block_weight):
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        pass

    def receive_blocks(self, json_dict):
        """
            PA Server receive a json_dict and send back a request
        :param json_dict:
        :return: 
        """
        pass


class PAServerCompack(PAClientComPack):

    def __init__(self, node, layer, block, content):

        PAClientComPack.__init__(self, node, layer, block, content)

    def compose_compack(content, params=None):

        return PAServerCompack(-1, -1, -1, content)

    def decompose_compack(cls, params=None):

        return PAServerCompack.from_dictionary(cls)


