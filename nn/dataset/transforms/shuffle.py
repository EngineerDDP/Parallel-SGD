import numpy as np

from nn.dataset.transforms.abstract import AbsTransformer


class Shuffle(AbsTransformer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "<Shuffle datasets.>"

    def run(self, train_x, train_y, test_x, test_y):
        index_train = [i for i in range(len(train_x))]
        np.random.shuffle(index_train)
        return train_x[index_train], train_y[index_train], test_x, test_y

    @property
    def params(self):
        return ()
