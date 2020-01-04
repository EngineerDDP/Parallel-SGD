import numpy as np

from codec.interfaces import ICommunicationCtrl, IComPack
from log import Logger
from profiles.settings import GlobalSettings

from codec.essential import BlockWeight, BatchWeight
from codec.ccdc import PartialBlockWeight, CodedBlockWeight


class NaiveDuplicationCodec(ICommunicationCtrl):
    """
        Full duplication coding strategy.
        Nodes that have the same block broadcast its block to all other nodes which
        do not having those blocks.
        Nodes only send parts of its block weights, the receiver must get all parts of
        data to restore the hole block weights.
    """

    def __init__(self, node_id, logger=Logger('NDC')):
        super().__init__()
        self.Node_ID = node_id

        self.block_weights_have = dict()
        self.partial_block_weights_buffer = dict()

    def dispose(self):
        self.block_weights_have.clear()
        self.partial_block_weights_buffer.clear()

    def update_blocks(self, block_weight:BlockWeight):
        block_weight = CodedBlockWeight.from_block_weight(block_weight)
        self.block_weights_have[block_weight.Block_ID] = block_weight
        send_to, compack = NaiveDuplicationComPack.compose_compack(self.Node_ID, block_weight)
        dic = compack.to_dictionary()

        # check for aggregate just in case there is only one working nodes.
        self.aggregate()
        yield (send_to, dic)

    def receive_blocks(self, json_dict:dict):
        pbw = NaiveDuplicationComPack.from_dictionary(json_dict).decompose_compack()
        self.partial_block_weights_buffer[(pbw.Block_ID, pbw.Position)] = pbw

        self.decoding(pbw)
        self.aggregate()

    def decoding(self, pbw:PartialBlockWeight):

        if len(self.partial_block_weights_buffer) < GlobalSettings.get_default().redundancy:
            return

        # search for all pieces
        search_ids = [(pbw.Block_ID, pos) for pos in range(GlobalSettings.get_default().redundancy)]
        search_results = []

        for id in search_ids:
            if self.partial_block_weights_buffer.get(id) is not None:
                search_results.append(self.partial_block_weights_buffer[id])
            else:
                return None

        search_results = sorted(search_results, key=lambda item: item.Position)
        partial_weights = [item.Content for item in search_results]

        result_weights = np.concatenate(partial_weights, axis=CodedBlockWeight.SPLIT_AXIS)

        self.block_weights_have[pbw.Block_ID] = BlockWeight(0, 0, pbw.Block_ID,
                set(GlobalSettings.get_default().block_assignment.block_2_node[pbw.Block_ID]), result_weights)

    def aggregate(self):
        if len(self.block_weights_have) == GlobalSettings.get_default().block_assignment.block_count:

            result_weights = 0
            # do aggregate
            for block_weights in self.block_weights_have.values():
                result_weights += block_weights.Content

            self.set_result(result_weights)
            self.dispose()

        return None


class NaiveDuplicationComPack(IComPack):

    def __init__(self, node_id, partial_block_weights_id, partial_block_weights_pos, partial_block_weights_content):

        self.Node_ID = node_id
        self.Partial_Block_Weights_ID = partial_block_weights_id
        self.Partial_Block_Weights_Pos = partial_block_weights_pos
        self.Partial_Block_Weights_Content = partial_block_weights_content

    def to_dictionary(self):
        dic = {
            "Node_ID": self.Node_ID,
            "Partial_Block_Weights_ID": self.Partial_Block_Weights_ID,
            "Partial_Block_Weights_Pos": self.Partial_Block_Weights_Pos,
            "Partial_Block_Weights_Content": self.Partial_Block_Weights_Content
        }
        return dic

    @staticmethod
    def from_dictionary(dic):
        node_id = dic["Node_ID"]
        id = dic["Partial_Block_Weights_ID"]
        pos = dic["Partial_Block_Weights_Pos"]
        content = dic["Partial_Block_Weights_Content"]

        return NaiveDuplicationComPack(node_id, id, pos, content)

    @staticmethod
    def compose_compack(node_id, block_weights=None):

        partial_block_weights = block_weights.get_by_node(node_id)
        send_to = block_weights.Adversary_ID

        compack = NaiveDuplicationComPack(node_id,
                                          partial_block_weights.Block_ID,
                                          partial_block_weights.Position,
                                          partial_block_weights.Content)

        return send_to, compack

    def decompose_compack(self, params=None):

        partial_block_weights = PartialBlockWeight(0, 0, self.Partial_Block_Weights_ID,
                                                   self.Partial_Block_Weights_Pos,
                                                   self.Partial_Block_Weights_Content)

        return partial_block_weights