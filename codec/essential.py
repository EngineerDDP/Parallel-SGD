class BatchWeight:

    def __init__(self, batch_id, layer_id, content):
        self.Batch_ID = batch_id
        self.Layer_ID = layer_id
        self.Content = content


class Block_Weight:
    """
        Weights calculated using one block
    """

    def __init__(self, layer_id, batch_id, block_id, company_id, adv_id, content):
        self.Layer_ID = layer_id
        self.Batch_ID = batch_id
        self.Block_ID = block_id
        self.Company_ID = company_id
        self.Adversary_ID = adv_id
        self.Content = content
