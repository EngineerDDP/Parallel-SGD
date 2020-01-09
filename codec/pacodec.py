from codec.interfaces import ICommunicationCtrl, IComPack, yield_none

from network.agreements import DefaultNodes
from log import Logger


class PAClientCodec(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('None')):

        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id
        self.Log = logger

    def dispose(self):
        """
            Nothing to do.
        """
        pass

    def update_blocks(self, block_weight):

        send, pack = PAClientComPack.compose_compack(block_weight, self.Node_ID)
        pack = PAClientComPack.to_dictionary(pack)
        yield (send, pack)

    def receive_blocks(self, json_dict):

        compack = PAServerCompack.from_dictionary(json_dict)
        self.set_result(compack.Content)

        return yield_none()


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

        send_target = [DefaultNodes.Parameter_Server]
        pack = PAClientComPack(node_id, block_weights.Layer_ID, block_weights.Block_ID, block_weights.Content)

        return send_target, pack

    def decompose_compack(self, params=None):
        raise NotImplementedError()


class PAServerCompack(PAClientComPack):

    def __init__(self, node, layer, block, content):

        PAClientComPack.__init__(self, node, layer, block, content)

    @staticmethod
    def compose_compack(content, params=None):
        return PAServerCompack(-1, -1, -1, content)

