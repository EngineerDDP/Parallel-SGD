from numpy import ndarray
from typing import Set


class BlockWeight:
    """
        Weights calculated using one block
    """

    def __init__(self, content: ndarray, block_id: int, company_ids: Set[int], adversary_ids: Set[int]):
        self.block_id = block_id
        self.company = company_ids
        self.adversary = adversary_ids
        self.content = content
