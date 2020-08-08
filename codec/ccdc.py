import numpy as np

# from threading import Lock
from threading import Lock
from itertools import combinations

from codec.essential import Block_Weight

from codec.interfaces import ICommunication_Ctrl
from codec.interfaces import IComPack, netEncapsulation

from profiles.settings import GlobalSettings
from utils.log import Logger


Qunatization = 'int32'
Scale = 100000000


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


class CodedBlockWeight(Block_Weight):

    SPLIT_AXIS = 0

    def __init__(self, layer_id, batch_id, block_id, company_id, content):
        Block_Weight.__init__(self, layer_id, batch_id, block_id, company_id, content)

    @classmethod
    def from_block_weight(cls, blockweight):
        return cls(blockweight.Layer_ID, blockweight.Batch_ID, blockweight.Block_ID, blockweight.Company_ID,
                   blockweight.Content)

    def __vsplit(self, content, count, take):
        # get position index
        pos = np.floor(np.linspace(0, content.shape[CodedBlockWeight.SPLIT_AXIS], count+1)).astype('int')
        return slice(pos[take], pos[take+1])

    def get_by_node(self, node_id):
        # get position
        pos = list(self.Company_ID).index(node_id)

        return self.get_by_position(pos)

    def get_by_position(self, pos):
        if CodedBlockWeight.SPLIT_AXIS == 1:
            parts = self.Content[:, self.__vsplit(self.Content, len(self.Company_ID), pos)].copy()
        else:
            # get parts
            parts = self.Content[self.__vsplit(self.Content, len(self.Company_ID), pos)].copy()

        # return value
        return PartialBlockWeight(self.Layer_ID, self.Batch_ID, self.Block_ID, pos, parts)


class CodedCommunicationCtrl(ICommunication_Ctrl):
    """
        Control communication between nodes within a specified batch and layer.

    """

    def __init__(self, node_id, logger=Logger('None')):
        ICommunication_Ctrl.__init__(self)

        self.Node_ID = node_id

        self.Total_Blocks = GlobalSettings.get_default().block_assignment.block_count

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

    def update_blocks(self, blockweight: Block_Weight):
        """
            Update a block weights to the cluster
        """
        # stabilize float
        blockweight.Content = (np.floor(blockweight.Content * Scale)).astype(Qunatization)
        self.block_weights_have[blockweight.Block_ID] = CodedBlockWeight.from_block_weight(blockweight)
        self.block_weights_recv[blockweight.Block_ID] = CodedBlockWeight.from_block_weight(blockweight)

        self.aggregate()
        # check if any of those blocks were ready to broadcast
        return self.coding()

    def receive_blocks(self, json_dict: dict):
        """
            Received a communication package from other node
        """
        # self.Log.print_log('Self have: blocks {}'.format(self.BlockWeights_Send.keys()))
        partial_block_weight = ComPack.from_dictionary(json_dict).decompose_compack(self.block_weights_have)

        self.Parts_BlockWeight_Buffers[
            (partial_block_weight.Block_ID, partial_block_weight.Position)] = partial_block_weight

        self.decoding(partial_block_weight.Block_ID)
        self.aggregate()

    def do_something_to_save_yourself(self):
        self.ComPack_Combs.clear()
        return self.coding()

    def coding(self):

        if len(self.block_weights_have) < GlobalSettings.get_default().redundancy:
            return None

        combs = combinations(self.block_weights_have.keys(), GlobalSettings.get_default().redundancy)

        for comb in combs:

            # check if sent
            if comb in self.ComPack_Combs:
                continue

            values = [self.block_weights_have[key] for key in comb]
            targets, compack = ComPack.compose_compack(self.Node_ID, values)
            self.ComPack_Combs.add(comb)
            dic = compack.to_dictionary()

            yield netEncapsulation(targets, dic)

    def decoding(self, new_block_id: int):

        if len(self.Parts_BlockWeight_Buffers) < GlobalSettings.get_default().redundancy:
            return

        search_index = [(new_block_id, pos) for pos in range(GlobalSettings.get_default().redundancy)]

        search_result = []

        for i in search_index:
            # if any of required blockweights were not available, give up search
            if self.Parts_BlockWeight_Buffers.get(i):
                search_result.append(self.Parts_BlockWeight_Buffers[i])
            else:
                return

        assert len(search_result) == GlobalSettings.get_default().redundancy, 'decode invaild'

        # retrieve info
        block_id = search_result[0].Block_ID

        # sort and concatenate
        partial_weight_array = sorted(search_result, key=lambda item: item.Position)
        partial_weight_array = [arr.Content for arr in partial_weight_array]
        result_weight = np.concatenate(partial_weight_array, axis=CodedBlockWeight.SPLIT_AXIS)

        # get layer id an batch id within this scope is deprecated
        # now using zero as default
        self.block_weights_recv[new_block_id] = CodedBlockWeight(0, 0, block_id,
             set(GlobalSettings.get_default().block_assignment.block_2_node[block_id]), result_weight)

        return

    def aggregate(self):

        if len(self.block_weights_recv) == self.Total_Blocks:

            block_weights = 0

            for i in self.block_weights_recv.values():
                block_weights += i.Content

            self.set_result((block_weights.astype('float64') / Scale))
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
        self.Partial_Block_Weights_Content = partial_block_weight_content
        self.Partial_Block_Weights_Content_Position = partial_block_weights_content_position

    def to_dictionary(self):

        dic = {
            'Node_ID': self.Node_ID,
            'Partial_Block_Weights_Content_ID': self.Partial_Block_Weights_Content_ID,
            'Partial_Block_Weights_Content_Position': self.Partial_Block_Weights_Content_Position,
            'Partial_Block_Weights_Content': self.Partial_Block_Weights_Content
        }

        return dic

    @staticmethod
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

    @staticmethod
    def compose_compack(node_id, block_weights=None):

        partial_block_weights_content_position = []
        partial_block_weights_content_id = []
        partial_block_weight_content = None
        send_target_prepare = {}
        send_target_set = []

        # validating package
        for block_weight in block_weights:
            for adv in block_weight.Adversary_ID:
                send_target_prepare[adv] = send_target_prepare.get(adv, 0) + 1

        # prepare send targets
        for key in send_target_prepare:
            if send_target_prepare[key] == 1:
                send_target_set.append(key)

        for block_weight in block_weights:
            # save block id
            partial_block_weights_content_id.append(block_weight.Block_ID)
            # get part of block weight
            partial_block_weight = block_weight.get_by_node(node_id)

            # initialization
            if partial_block_weight_content is None:
                partial_block_weight_content = partial_block_weight.Content
            else:
                # save content and position
                partial_block_weight_content ^= partial_block_weight.Content
            partial_block_weights_content_position.append(partial_block_weight.Position)

        pack = ComPack(node_id, partial_block_weights_content_id,
                       partial_block_weights_content_position, partial_block_weight_content)

        return send_target_set, pack

    def decompose_compack(self, block_weights_dic=None):

        iteration = zip(self.Partial_Block_Weights_Content_ID,
                        self.Partial_Block_Weights_Content_Position)
        content = self.Partial_Block_Weights_Content

        # deprecated usage
        layer_id = list(block_weights_dic.values())[0].Layer_ID
        batch_id = list(block_weights_dic.values())[0].Batch_ID

        parts_absent = 0
        decompose_part_id = 0
        decompose_part_pos = 0

        for id, pos in iteration:
            if block_weights_dic.get(id):
                content ^= block_weights_dic[id].get_by_position(pos).Content
            else:
                parts_absent += 1
                decompose_part_id = id
                decompose_part_pos = pos

        assert parts_absent == 1, 'Invalid decode process, value absent: {}'.format(parts_absent)

        return PartialBlockWeight(layer_id, batch_id, decompose_part_id, decompose_part_pos, content)


if __name__ == '__main__':
    bw = Block_Weight(0, 0, 0, {0, 1}, [1, 2, 3])
    cbw = CodedBlockWeight.from_block_weight(bw)
    print(cbw)
