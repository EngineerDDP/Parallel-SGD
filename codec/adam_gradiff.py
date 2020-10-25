import numpy as np

from typing import Union, Dict, Iterable, Tuple

from nn.gradient_descent import ADAMOptimizer

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation


class ADAMParameterServer(Codec):

    def __init__(self, node_id):

        Codec.__init__(self, node_id)
        # save PA current state
        self.__current_weights: Union[np.ndarray, float] = 0
        # save previous state for each node
        self.__bak_weights_node: Dict[int, Union[np.ndarray, float]] = {}
        # Adam optimizer
        self.__adam = ADAMOptimizer()

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server Cannot update blocks!
        :param block_weight:
        :return:
        """
        pass

    def receive_blocks(self, content: Tuple[int, np.ndarray]) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        """
            PA Server receive a json_dict and send back a request
        :param content:
        :return:
        """
        # get last state of working node
        last_state = self.__bak_weights_node.get(content[0], 0)
        delta = self.__adam.delta(content[1])
        # update global current state
        self.__current_weights = self.__current_weights + delta
        # get difference
        grad_diff = self.__current_weights - last_state
        # update last state of working node
        self.__bak_weights_node[content[0]] = self.__current_weights

        return netEncapsulation(content[0], (-1, grad_diff))
