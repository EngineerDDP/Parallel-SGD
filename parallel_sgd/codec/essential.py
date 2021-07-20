from numpy import ndarray
from typing import Set


class BlockWeight:
    """
        Weights calculated using one block
    """

    def __init__(self, content: ndarray, block_id: int):
        self.block_id = block_id
        self.content = content
