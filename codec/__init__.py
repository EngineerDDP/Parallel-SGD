from profiles import Settings

class GlobalSettings:
    deprecated_default_settings = None

    @staticmethod
    def get_default() -> Settings:
        return GlobalSettings.deprecated_default_settings


class Tag:

    def __init__(self, batch, block, node, companies, all, layer=0, batch_no=0):
        """
            Note: Companies including itself
        """

        self.Layer_No = layer
        self.Batch_No = batch_no
        self.Batch = batch
        self.Block_No = block
        self.Node_No = node
        self.Company = companies
        self.Adversary = all - companies

    # used in backward propagation
    def incBatch(self):
        self.Batch_No += 1
        self.resetLayer()

    def incLayer(self):
        self.Layer_No += 1

    def resetLayer(self):
        self.Layer_No = 0

    def getSlice(self):
        # which batch
        offset = self.Batch_No * self.Batch.Batch_Size
        # get slice
        sli = self.Batch.get_slice(offset, self.Block_No)
        # return slice
        return sli

    def getSliceWithinBatch(self):
        # get slice
        sli = self.Batch.get_slice(0, self.Block_No)
        # return slice
        return sli

    def copy(self):
        return Tag(self.Batch, self.Block_No, self.Node_No, self.Company, int(self.Layer_No), int(self.Batch_No))


def build_tags(node_id: int, settings:Settings):
    if not isinstance(node_id, int):
        node_id = int(node_id)

    assert node_id < settings.node_count, "This worker has nothing to do."

    batch = settings.batch
    blocks = settings.block_assignment.node_2_block[int(node_id)]
    nodes = settings.block_assignment.block_2_node
    tags = [Tag(batch, block, node_id, set(nodes[block]), settings.nodes) for block in blocks]

    return tags
