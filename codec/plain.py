from codec.essential import BlockWeight
from codec.interfaces import ICommunicationCtrl, IComPack

from profiles.settings import GlobalSettings
from log import Logger


class PlainCommunicationCtrl(ICommunicationCtrl):

    def __init__(self, node_id, logger=Logger('None')):

        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id

        self.BlockWeights = dict()

        # system diagnosed info
        self.Log = logger

        # assert GlobalSettings.get_default().Redundancy == 1, "Uncoded asgd cant run in redundancy mode"

    def dispose(self):
        """
            Dispose this object
        :return: None
        """
        self.BlockWeights.clear()
        self.Log.log_message('Dispose garbage complete, Node:{}'.format(self.Node_ID))

    def update_blocks(self, blockweight):
        """
            return what to send
        :param blockweight: block weights just computed
        :return: the target node id , and the data pack to be sent
        """

        self.BlockWeights[blockweight.Block_ID] = blockweight

        send, pack = PlainComPack.compose_compack(self.Node_ID, blockweight)
        pack = pack.to_dictionary()

        self.check_for_combine(blockweight.Block_ID)

        yield (send, pack)

    def receive_blocks(self, json_dict):

        block_weight = PlainComPack.from_dictionary(json_dict).decompose_compack()
        self.BlockWeights[block_weight.Block_ID] = block_weight

        self.check_for_combine(block_weight.Block_ID)

    def check_for_combine(self, new_block_id):

        if len(self.BlockWeights) < GlobalSettings.get_default().block_assignment.block_count:
            return

        batchweight = 0

        current_values = self.BlockWeights.values()
        for i in current_values:
            batchweight += i.Content

        self.set_result(batchweight)
        self.BlockWeights.clear()

        return


class PlainComPack(IComPack):

    def __init__(self, node_id, layer_id, batch_id, content):
        self.Node_ID = node_id
        self.Layer_ID = layer_id
        self.Batch_ID = batch_id

        self.Content = content

    def to_dictionary(self):
        dic = {
            'Node_ID':  self.Node_ID,
            'Layer_ID': self.Layer_ID,
            'Batch_ID': self.Batch_ID,
            'Content':  self.Content
        }

        return dic

    @staticmethod
    def from_dictionary(dic):
        node_id = dic['Node_ID']
        layer_id = dic['Layer_ID']
        batch_id = dic['Batch_ID']
        content = dic['Content']

        pac = PlainComPack(node_id, layer_id, batch_id, content)

        return pac

    @staticmethod
    def compose_compack(node_id, block_weight=None):
        send_target = GlobalSettings.get_default().nodes - {node_id}

        pack = PlainComPack(node_id, block_weight.Layer_ID, block_weight.Batch_ID, block_weight.Content)

        return send_target, pack

    def decompose_compack(self, params=None):
        block_id = GlobalSettings.get_default().block_assignment.node_2_block[self.Node_ID]
        content = self.Content

        blockweight = BlockWeight(self.Layer_ID, self.Batch_ID, block_id[0], set(block_id), content)

        return blockweight