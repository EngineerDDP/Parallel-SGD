import numpy as np

# from threading import Lock
from threading import Lock
from itertools import combinations

from codec.essential import BlockWeight
from codec.essential import BatchWeight

from codec.interfaces import ICommunicationCtrl
from codec.interfaces import IComPack

from settings import GlobalSettings
from log import Logger


class PartialBlockWeight:
    """
        Weights calculated using one block and prepared to be coded
    """

    def __init__(self, layer_id, batch_id, block_id, position, content):
        self.Layer_ID = layer_id
        self.Batch_ID = batch_id
        self.Block_ID = block_id
        self.Position = position
        self.Content = content


class CodedBlockWeight(BlockWeight):

    def __init__(self, layer_id, batch_id, block_id, company_id, content):
        BlockWeight.__init__(self, layer_id, batch_id, block_id, company_id, content)

    @classmethod
    def fromBlockWeight(cls, blockweight):
        return cls(blockweight.Layer_ID, blockweight.Batch_ID, blockweight.Block_ID, blockweight.Company_ID,
                   blockweight.Content)

    def getbyNode(self, node_id):
        # get position
        pos = list(self.Company_ID).index(node_id)

        # get parts
        parts = np.vsplit(self.Content, len(self.Company_ID))

        # return value
        return PartialBlockWeight(self.Layer_ID, self.Batch_ID, self.Block_ID, pos, parts[pos])

    def getbyPosition(self, pos):
        # get parts
        parts = np.vsplit(self.Content, len(self.Company_ID))

        # return value
        return PartialBlockWeight(self.Layer_ID, self.Batch_ID, self.Block_ID, pos, parts[pos])


class CodedCommunicationCtrl(ICommunicationCtrl):
    """
        Control communication between nodes within a specified batch and layer
    """

    def __init__(self, node_id, logger=Logger('None')):
        ICommunicationCtrl.__init__(self)

        self.Node_ID = node_id

        self.Total_Blocks = GlobalSettings.getDefault().BlockCount

        # the block weights can be calculated locally
        self.block_weights_have = dict()
        # the block weights can be received remotely
        self.block_weights_recv = dict()

        # Partial block weights
        # (Block_ID, Pos) -> PartialBlockWeight
        self.Parts_BlockWeight_Buffers = {}

        # combination tuple type
        self.ComPack_Combs = set()

        # thread access control
        self.Locker = Lock()

        # diagnosed info logger
        self.Log = logger

    def dispose(self):
        """
            dispose this object and free all the memory allocated.
        """
        self.block_weights_have.clear()
        self.block_weights_recv.clear()
        self.Parts_BlockWeight_Buffers.clear()
        self.ComPack_Combs.clear()

    def update_blocks(self, blockweight):
        """
            Update a block weights to the cluster
        """
        # stabilize float
        blockweight.Content = np.floor(blockweight.Content * 1000) / 1000

        self.block_weights_have[blockweight.Block_ID] = CodedBlockWeight.fromBlockWeight(blockweight)
        self.block_weights_recv[blockweight.Block_ID] = CodedBlockWeight.fromBlockWeight(blockweight)

        # self.aggregate()
        # check if any of those blocks were ready to broadcast
        return self.coding()

    def receive_blocks(self, json_dict):
        """
            Received a communication package from other node
        """
        # self.Log.print_log('Self have: blocks {}'.format(self.BlockWeights_Send.keys()))
        partial_block_weight = ComPack.decompose_compack(ComPack.from_dictionary(json_dict), self.block_weights_have)

        self.Parts_BlockWeight_Buffers[
            (partial_block_weight.Block_ID, partial_block_weight.Position)] = partial_block_weight

        self.decoding(partial_block_weight.Block_ID)
        self.aggregate()

    def coding(self):

        if len(self.block_weights_have) < GlobalSettings.getDefault().Redundancy:
            return None

        combs = combinations(self.block_weights_have.keys(), GlobalSettings.getDefault().Redundancy)

        for comb in combs:

            # check if sent
            if comb in self.ComPack_Combs:
                continue

            values = [self.block_weights_have[key] for key in comb]
            targets, compack = ComPack.compose_compack(self.Node_ID, values)
            self.ComPack_Combs.add(comb)
            compack = ComPack.to_dictionary(compack)

            yield (targets, compack)

    def decoding(self, new_block_id):

        if len(self.Parts_BlockWeight_Buffers) < GlobalSettings.getDefault().Redundancy:
            return

        search_index = [(new_block_id, pos) for pos in range(GlobalSettings.getDefault().Redundancy)]

        search_result = []

        for i in search_index:
            # if any of required blockweights were not available, give up search
            if self.Parts_BlockWeight_Buffers.get(i):
                search_result.append(self.Parts_BlockWeight_Buffers[i])
            else:
                return

        assert len(search_result) == GlobalSettings.getDefault().Redundancy, 'decode invaild'

        # retrieve info
        block_id = search_result[0].Block_ID

        # sort and concatenate
        partial_weight_array = sorted(search_result, key=lambda item: item.Position)
        partial_weight_array = [arr.Content for arr in partial_weight_array]
        result_weight = np.concatenate(partial_weight_array, axis=0)

        # get layer id an batch id within this scope is deprecated
        # now using zero as default
        self.block_weights_recv[new_block_id] = CodedBlockWeight(0, 0, block_id,
                                set(GlobalSettings.getDefault().BlockAssignment.Block2Node[block_id]), result_weight)

        return

    def aggregate(self):

        if len(self.block_weights_recv) == self.Total_Blocks:

            layer = 0
            batch = 0
            blockweights = 0

            for i in self.block_weights_recv.values():
                blockweights += i.Content
                layer = i.Layer_ID
                batch = i.Batch_ID

            self.set_result(blockweights)
            self.dispose()

        return None


class ComPack(IComPack):
    """
        CCDC Coding schema
    """

    def __init__(self, node_id, partial_block_weights_content_id,
                 partial_block_weights_content_position, partial_block_weight_content):

        self.Node_ID = node_id

        self.Partial_Block_Weights_Content_ID = partial_block_weights_content_id
        self.Partial_Block_Weights_Content = np.asarray(partial_block_weight_content, dtype='float16')
        self.Partial_Block_Weights_Content_Position = partial_block_weights_content_position

    def to_dictionary(compack):

        dic = {
            'Node_ID': compack.Node_ID,
            'Partial_Block_Weights_Content_ID': compack.Partial_Block_Weights_Content_ID,
            'Partial_Block_Weights_Content_Position': compack.Partial_Block_Weights_Content_Position,
            'Partial_Block_Weights_Content': compack.Partial_Block_Weights_Content
        }

        return dic

    def from_dictionary(dic):

        node_id = dic['Node_ID']
        partial_block_weights_content_id = dic['Partial_Block_Weights_Content_ID']
        partial_block_weights_content_position = dic['Partial_Block_Weights_Content_Position']
        partial_block_weight_content = dic['Partial_Block_Weights_Content']

        pac = ComPack(node_id,
                      partial_block_weights_content_id,
                      partial_block_weights_content_position,
                      partial_block_weight_content)

        return pac

    def compose_compack(node_id, blockweights=None):

        partial_block_weights_content_position = []
        partial_block_weights_content_id = []
        partial_block_weight_content = 0
        send_target_set = set()

        for blockweight in blockweights:
            # save block id
            partial_block_weights_content_id.append(blockweight.Block_ID)
            # get part of block weight
            partial_block_weight = blockweight.getbyNode(node_id)
            # save content and position
            partial_block_weight_content += partial_block_weight.Content
            partial_block_weights_content_position.append(partial_block_weight.Position)
            # get send target
            send_target_set ^= blockweight.Adversary_ID

        pack = ComPack(node_id, partial_block_weights_content_id,
                       partial_block_weights_content_position, partial_block_weight_content)

        return send_target_set, pack

    def decompose_compack(com_pack, blockweights_dic=None):

        iteration = zip(com_pack.Partial_Block_Weights_Content_ID,
                        com_pack.Partial_Block_Weights_Content_Position)
        content = com_pack.Partial_Block_Weights_Content

        # deprecated usage
        layer_id = list(blockweights_dic.values())[0].Layer_ID
        batch_id = list(blockweights_dic.values())[0].Batch_ID

        parts_absent = 0
        decompose_part_id = 0
        decompose_part_pos = 0

        for id, pos in iteration:
            if blockweights_dic.get(id):
                content -= blockweights_dic[id].getbyPosition(pos).Content
            else:
                parts_absent += 1
                decompose_part_id = id
                decompose_part_pos = pos

        assert parts_absent == 1, 'Invalid decode process, value absent: {}'.format(parts_absent)

        return PartialBlockWeight(layer_id, batch_id, decompose_part_id, decompose_part_pos, content)


if __name__ == '__main__':
    bw = BlockWeight(0, 0, 0, {0, 1}, [1, 2, 3])
    cbw = CodedBlockWeight.fromBlockWeight(bw)
    print(cbw)