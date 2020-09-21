from numpy import ndarray

from nn.data.interface import IDataFeeder


class NumpyDataFeeder(IDataFeeder):

    def __init__(self, x:ndarray, y:ndarray, batch_size:int):
        self.__iter_x:ndarray = x
        self.__iter_y:ndarray = y
        self.__batch_size = min(batch_size, len(x))
        self.__batches = len(x) // batch_size
        self.__iter = 0

    @property
    def position(self):
        return self.__iter

    @property
    def length(self):
        return self.__batches

    def __iter__(self):
        for self.__iter in range(self.__batches):
            start = self.__iter * self.__batch_size % (len(self.__iter_x) - self.__batch_size + 1)
            end = start + self.__batch_size
            part_x = self.__iter_x[start:end]
            part_y = self.__iter_y[start:end]
            self.__iter += 1
            yield part_x, part_y

    def __repr__(self):
        print("<Numpy data iterator, current batch: {}, total: {}.".format(self.__iter, self.__batches))