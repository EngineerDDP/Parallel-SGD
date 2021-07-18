from parallel_sgd.codec.interfaces import Codec
from parallel_sgd.codec.essential import BlockWeight


class DummyCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        self.set_result(block_weight.content)

    def receive_blocks(self, json_dict: dict):
        pass
