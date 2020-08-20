import numpy as np


def make_non_iid_distribution(x, y, batch_size):
    """
        Make your dataset i.i.d. compatible.
        Transform input x, y into non-i.i.d. data distribution.
    :param x: input samples
    :param y: input labels
    :param batch_size: batch_size while splitting.
    :return: (x, y) with non-i.i.d. distribution
    """
    # get total sample count
    n = y.shape[0]
    # get real n
    n = n - n % batch_size

    # get margin for each sampling point
    margin = n // batch_size
    # get batch sampling point, each batch corresponds with a column.
    indicator = np.arange(0, n).reshape([batch_size, margin])
    # transpose and reshape indicator
    indicator = np.reshape(indicator.T, newshape=-1)
    # get sorts index
    idx = np.argsort(y)
    # sampling data index according to sampling indicator
    idx = idx[indicator]
    # sampling read data
    return x[idx], y[idx]


if __name__ == '__main__':
    from dataset.simdata import load
    x, y, _, _ = load()
    x, y = make_non_iid_distribution(x, y, 64)
    print(y[:128])