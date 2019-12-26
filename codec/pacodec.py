from codec.essential import BatchWeight
from codec.essential import BlockWeight
from codec.interfaces import ICommunicationCtrl, IComPack

from network.agreements import DefaultNodes

from server_util.init_model import ModelMNIST
from settings import GlobalSettings

from log import Logger


def yield_none():
    # for iteration
    for i in []:
        yield ([], None)


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

        compack = PAServerCompack.decompose_compack(json_dict)
        self.set_result(compack.Content)

        return yield_none()


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

    def compose_compack(blockweights, node_id=None):

        send_target = [DefaultNodes.Parameter_Server]
        pack = PAClientComPack(node_id, blockweights.Layer_ID, blockweights.Block_ID, blockweights.Content)

        return send_target, pack

    def decompose_compack(cls, params=None):

        return PAClientComPack.from_dictionary(cls)


class PAServerCodec(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('PAS')):

        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id
        self.Learn_Rate = ModelMNIST.learn_rate()

        # save PA current state
        self.Current_Weights = 0

        self.Log = logger

        # save previous state for each node
        self.Bak_Weights_Node = {}
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
        return yield_none()

    def receive_blocks(self, json_dict):
        """
            PA Server receive a json_dict and send back a request
        :param json_dict:
        :return: 
        """
        # analyze received data
        compack = PAClientComPack.decompose_compack(json_dict)
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


class PAServerCompack(PAClientComPack):

    def __init__(self, node, layer, block, content):

        PAClientComPack.__init__(self, node, layer, block, content)

    def compose_compack(content, params=None):

        return PAServerCompack(-1, -1, -1, content)

    def decompose_compack(cls, params=None):

        return PAServerCompack.from_dictionary(cls)


