import numpy as np

from typing import Dict, Optional, Iterable, Tuple, Union
from numpy import ndarray

from codec import GlobalSettings
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation


class Plain(Codec):

    def __init__(self, node_id):

        super().__init__(node_id)
        self.BlockWeights: Dict[int, ndarray] = dict()

    def dispose(self):
        """
            Dispose this object
        :return: None
        """
        self.BlockWeights.clear()

    def update_blocks(self, block_weight: BlockWeight) -> netEncapsulation[Tuple[int, ndarray]]:
        """
            Try collect all blocks.
        """
        self.BlockWeights[block_weight.block_id] = block_weight.content
        self.check_for_combine()
        send_to = GlobalSettings.get_default().get_adversary(block_weight.block_id)
        return netEncapsulation(send_to, (block_weight.block_id, block_weight.content))

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        """
            Try collect all blocks.
        """
        self.BlockWeights[content[0]] = content[1]
        self.check_for_combine()

    def check_for_combine(self):

        if len(self.BlockWeights) < GlobalSettings.get_default().block_count:
            return

        res = 0
        for val in self.BlockWeights.values():
            res += val
        self.set_result(res / len(self.BlockWeights))
        self.BlockWeights.clear()
