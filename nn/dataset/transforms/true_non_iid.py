import numpy as np

from nn.dataset.transforms.abstract import AbsTransformer


class True_Non_IID(AbsTransformer):
    """
        Make your dataset i.i.d. compatible.
        Transform input_ref x, y into non-i.i.d. data distribution.

    :param batch_size: batch_size while splitting.
    :param disorder: non-iid has disorder rate, The higher the disorder, the more likely shuffle.
    :return: (x, y) with non-i.i.d. distribution
    """

    def __init__(self, batch_size, disorder):
        super().__init__()
        self.__batch_size = batch_size
        self.__disorder = disorder

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
        t_x,t_y = x[idx], y[idx]

        chaos_num = int(n * self.__disorder)
        index_train = [i for i in range(chaos_num)]
        np.random.shuffle(index_train)
        t_x_0 = t_x[:chaos_num]
        t_y_0 = t_y[:chaos_num]
        t_x_0 = t_x_0[index_train]
        t_y_0 = t_y_0[index_train]
        t_x = np.concatenate([t_x_0,t_x[chaos_num:]], axis=0)
        t_y = np.concatenate([t_y_0,t_y[chaos_num:]], axis=0)

        return t_x, t_y, test_x, test_y


if __name__ == '__main__':
    from nn import dataset as cifar

    mk = Make_Non_IID(64, 0.5)
    x,y,_,_ = mk.run(*cifar.CIFAR().load())
