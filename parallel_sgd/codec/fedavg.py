from typing import Iterable, Tuple, Union, Dict

import numpy as np
from numpy import ndarray

from parallel_sgd.codec import GlobalSettings
from parallel_sgd.codec.essential import BlockWeight
from parallel_sgd.codec.interfaces import Codec, netEncapsulation
from constants import Parameter_Server


# 本文件力求还原最土味的最原始的FEDAVG
class FedAvgClient(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.__local_turn = 0
        self.__TURN = 150

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        self.__local_turn += 1
        if self.__local_turn >= self.__TURN:
            return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))
        else:
            self.set_result(block_weight.content)

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        self.__local_turn = 0
        self.set_result(content[1])


class FedAvgServer(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)
        self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}

    def dispose(self):
        self.Bak_Weights_Node.clear()

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        pass

    def receive_blocks(self, content: Tuple[int, ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server receive a json_dict and send back a request
        :param content:
        :return:
        """
        # update global current state
        self.Bak_Weights_Node[content[0]] = content[1]
        if len(self.Bak_Weights_Node) == GlobalSettings.get_default().node_count:
            global_weight = np.mean(list(self.Bak_Weights_Node.values()), axis=0)
            self.dispose()
            return netEncapsulation(GlobalSettings.get_default().nodes, (Parameter_Server, global_weight))
