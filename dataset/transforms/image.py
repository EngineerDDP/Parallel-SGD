import numpy as np

from dataset.transforms.__init__ import AbsTransformer


class ImageCls(AbsTransformer):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "<Scale to 1.0 and make onehot label>"

    @property
    def params(self):
        return tuple()

    def run(self, train_x, train_y, test_x, test_y) -> tuple:
        return train_x / 255, np.eye(10)[train_y], test_x / 255, np.eye(10)[test_y]