import numpy as np

from typing import Dict, Optional, Iterable, Tuple, Union
from numpy import ndarray

from codec import GlobalSettings
from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation


class PlainCommunicationCtrl(Codec):

    def __init__(self, node_id):

        super().__init__(node_id)
        self.BlockWeights: Dict[int, ndarray] = dict()

    def dispose(self):
        """
            Dispose this object
        :return: None
        """
        self.BlockWeights.clear()

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            Try collect all blocks.
        """
        self.BlockWeights[block_weight.block_id] = block_weight.content
        self.check_for_combine()
        return netEncapsulation(block_weight.adversary, (block_weight.block_id, block_weight.content))

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        """
            Try collect all blocks.
        """
        self.BlockWeights[content[0]] = content[1]
        self.check_for_combine()

    def check_for_combine(self):

        if len(self.BlockWeights) < GlobalSettings.get_default().block_count:
            return

        self.set_result(np.sum(self.BlockWeights.values()))
        self.BlockWeights.clear()
