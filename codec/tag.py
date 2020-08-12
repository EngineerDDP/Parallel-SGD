class Tag:

    def __init__(self, batch, block, node, companies, layer=0, batch_no=0):
        """
            Note: Companies including itself
        """

        self.Layer_No = layer
        self.Batch_No = batch_no
        self.Batch = batch
        self.Block_No = block
        self.Node_No = node
        self.Company = companies

    def incBatch(self):
        self.Batch_No += 1
        self.Layer_No = 0

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

