from codec.essential import BatchWeight
from codec.essential import BlockWeight
from codec.interfaces import ICommunicationCtrl, IComPack

from utils.delegate import Delegation

from settings import GlobalSettings

from log import Logger


class PlainCommunicationCtrl(ICommunicationCtrl):

    def __init__(self, node_id, batch_id, layer_id, logger=Logger('None')):

        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id
        self.Batch_ID = batch_id
        self.Layer_ID = layer_id

        self.BlockWeights = dict()

        # system diagnosed info
        self.Log = logger

        assert GlobalSettings.getDefault().Redundancy == 1, "Uncoded asgd cant run in redundancy mode"

    def dispose(self):
        """
            Dispose this object
        :return: None
        """
        self.BlockWeights.clear()
        self.Log.log_message('Dispose garbage complete, Batch: {}, Layer: {}'.format(self.Batch_ID, self.Layer_ID))

    def update_blocks(self, blockweight):
        """
            return what to send
        :param blockweight: block weights just computed
        :return: the target node id , and the data pack to be sent
        """

        self.BlockWeights[blockweight.Block_ID] = blockweight

        send, pack = PlainComPack.compose_compack(self.Node_ID, blockweight)
        pack = PlainComPack.to_dictionary(pack)

        self.check_for_combine(blockweight.Block_ID)

        return send, pack

    def receive_blocks(self, json_dict):

        blockweight = PlainComPack.decompose_compack(PlainComPack.from_dictionary(json_dict))
        self.BlockWeights[blockweight.Block_ID] = blockweight

        self.Log.log_message('Received new block, Batch: {}, Layer: {}, Block: {}.'
                           .format(blockweight.Batch_ID, blockweight.Layer_ID, blockweight.Block_ID))

        self.check_for_combine(blockweight.Block_ID)

    def check_for_combine(self, new_block_id):

        if len(self.BlockWeights) < GlobalSettings.getDefault().BlockCount:
            return

        batchweight = 0

        current_values = self.BlockWeights.values()
        for i in current_values:
            batchweight += i.Content

        self.updated_weight_buffer += batchweight

        return


class PlainComPack(IComPack):

    def __init__(self, node_id, layer_id, batch_id, content):
        self.Node_ID = node_id
        self.Layer_ID = layer_id
        self.Batch_ID = batch_id

        self.Content = content

    def to_dictionary(compack):
        dic = {
            'Node_ID': compack.Node_ID,
            'Layer_ID': compack.Layer_ID,
            'Batch_ID': compack.Batch_ID,
            'Content': compack.Content
        }

        return dic

    def from_dictionary(dic):
        node_id = dic['Node_ID']
        layer_id = dic['Layer_ID']
        batch_id = dic['Batch_ID']
        content = dic['Content']

        pac = PlainComPack(node_id, layer_id, batch_id, content)

        return pac

    def compose_compack(node_id, blockweight):
        send_target = GlobalSettings.getDefault().Nodes - {node_id}

        pack = PlainComPack(node_id, blockweight.Layer_ID, blockweight.Batch_ID, blockweight.Content)

        return send_target, pack

    def decompose_compack(com_pack):
        block_id = GlobalSettings.getDefault().BlockAssignment.Node2Block[com_pack.Node_ID]
        content = com_pack.Content

        blockweight = BlockWeight(com_pack.Layer_ID, com_pack.Batch_ID, block_id[0], set(block_id), content)

        return blockweight