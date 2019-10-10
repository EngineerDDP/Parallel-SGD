from server_util.codec.interfaces import PACodec


class PlainParameterServerCodec(PACodec):

    def __init__(self, initial_weights, layer_id, learn_rate):

        # initialize weights state @ t = 0
        self.Weights_Content = initial_weights

        # Other parameters
        self.Layer_ID = layer_id
        self.Learn_Rate = learn_rate

    def update_blocks(self, content, tag):

        self.Weights_Content -= self.Learn_Rate * content

    def acquire_weights(self, tag):

        return self.Weights_Content


