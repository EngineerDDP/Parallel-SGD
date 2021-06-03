from typing import Optional, Iterable, Tuple, Union, Dict

from numpy import ndarray

from codec.essential import BlockWeight
from codec.interfaces import Codec, netEncapsulation
from constants import Parameter_Server


class PSClient(Codec):

    def __init__(self, node_id):
        Codec.__init__(self, node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight) -> Union[Iterable[netEncapsulation], netEncapsulation, None]:
        return netEncapsulation(Parameter_Server, (self.node_id, block_weight.content))

    def receive_blocks(self, content: Tuple[int, ndarray]) -> None:
        self.set_result(content[1])


class GradDiffPS(Codec):

    def __init__(self, node_id):

        Codec.__init__(self, node_id)
        # save PA current state
        self.Current_Weights: Union[ndarray, float] = 0
        # save previous state for each node
        self.Bak_Weights_Node: Dict[int, Union[ndarray, float]] = {}

    def dispose(self):
        pass

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
        # get last state of working node
        last_state = self.Bak_Weights_Node.get(content[0], 0)
        # update global current state
        self.Current_Weights = self.Current_Weights + content[1]
        # get difference
        grad_diff = self.Current_Weights - last_state
        # update last state of working node
        self.Bak_Weights_Node[content[0]] = self.Current_Weights

        return netEncapsulation(content[0], (-1, grad_diff))


class ParaAveragingServer(Codec):

    def __init__(self, node_id):

        Codec.__init__(self, node_id)
        self.Current_Weights: Optional[float, ndarray] = 0

    def dispose(self):
        pass

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
        self.Current_Weights = self.Current_Weights - content[1]

        return netEncapsulation(content[0], (-1, self.Current_Weights))
