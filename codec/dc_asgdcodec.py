from typing import Iterable, Optional, Tuple, Dict, Union

import numpy as np

from codec.interfaces import netEncapsulation
from codec.naive_ps import GradDiffPS


class DCASGDServerCodec(GradDiffPS):
    """
        Implemented according to paper:
        Shuxin Zheng, Qi Meng, Taifeng Wang, et al. Asynchronous Stochastic Gradient Descent with Delay Compensation.
        International Conference on Machine Learning (ICML), Sydney, Australia. 2017.
    """

    def __init__(self, node_id: int):

        super().__init__(node_id)

        # init weights
        self.Weights_init = 0

        # other parameters
        self.Learn_Rate = 1
        self.Bak_Weights_Node: Dict[int, np.ndarray] = {}
        self.Lambda_T = 2
        self.Mean_Square = 0
        self.Mean_Square_Epsilon = 1e-7
        self.Mean_Square_M = 0.95

    def receive_blocks(self, content: Tuple[int, np.ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation]:
        """
            Adaptive DC-ASGD algorithm.
        """
        content_square = np.multiply(content[1], content[1])
        # Update weights with delay-compensation
        delay = np.multiply(content_square, self.Weights_init - self.Bak_Weights_Node[content[0]])
        self.Weights_init = self.Weights_init - self.Learn_Rate * (content[1] + self.Lambda_T * delay)
        self.Mean_Square = self.Mean_Square_M * self.Mean_Square + (1 - self.Mean_Square_M) * content_square
        self.Lambda_T = self.Lambda_T / np.sqrt(self.Mean_Square + self.Mean_Square_Epsilon)
        # Send back updated weights
        self.Bak_Weights_Node[content[0]] = self.Weights_init
        return netEncapsulation(content[0], (-1, self.Weights_init))
