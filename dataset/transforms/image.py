import numpy as np

from dataset.transforms.__init__ import AbsTransformer


class ImageCls(AbsTransformer):

    def __init__(self):
        super().__init__()

    def params(self):
        return tuple()

    def run(self, train_x, train_y, test_x, test_y) -> tuple:
        return train_x / 255, np.eye(10)[train_y], test_x / 255, np.eye(10)[test_y]