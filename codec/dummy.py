from codec import GlobalSettings
from codec.interfaces import Codec
from codec.essential import BlockWeight
from codec.interfaces import netEncapsulation


class DummyCodec(Codec):

    def __init__(self, node_id):
        super().__init__(node_id)

    def dispose(self):
        pass

    def update_blocks(self, block_weight: BlockWeight):
        self.set_result(block_weight.content)

    def receive_blocks(self, json_dict: dict):
        pass
