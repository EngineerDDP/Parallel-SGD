import numpy as np

from nn.dataset.transforms.abstract import AbsTransformer


class Make_Non_IID(AbsTransformer):
    """
        Make your dataset i.i.d. compatible.
        Transform input_ref x, y into non-i.i.d. data distribution.
    :param x: input_ref samples
    :param y: input_ref labels
    :param batch_size: batch_size while splitting.
    :return: (x, y) with non-i.i.d. distribution
    """

    def __init__(self, batch_size):
        super().__init__()
        self.__batch_size = batch_size

    def __repr__(self):
        return "<Make non-iid dataset, based on labels>"

    @property
    def params(self):
        return self.__batch_size,

    def run(self, x, y, test_x, test_y) -> tuple:
        # get total sample count
        n = y.shape[0]
        # get real n
        n = n - n % self.__batch_size

        # get margin for each sampling point
        margin = n // self.__batch_size
        # get batch sampling point, each batch corresponds with a column.
        indicator = np.arange(0, n).reshape([self.__batch_size, margin])
        # transpose and reshape indicator
        indicator = np.reshape(indicator.T, newshape=-1)
        # get sorts index
        idx = np.argsort(y)
        # sampling data index according to sampling indicator
        idx = idx[indicator]
        # sampling read data
        return x[idx], y[idx], test_x, test_y
