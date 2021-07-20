import struct
import numpy as np
import os
import hashlib

from nn.dataset.interfaces import AbsDataset


class MNIST(AbsDataset):

    def __init__(self, check_sum=None):
        self.path = './.data/mnist_data/'
        super().__init__(check_sum)

    def __repr__(self):
        return '<MNIST classification dataset.>'

    def check_sum(self) -> str:
        if not (os.path.exists(os.path.join(self.path, '%s-labels.idx1-ubyte' % 'train'))
            and os.path.exists(os.path.join(self.path, '%s-labels.idx1-ubyte' % 't10k'))
            and os.path.exists(os.path.join(self.path, '%s-images.idx3-ubyte' % 'train'))
            and os.path.exists(os.path.join(self.path, '%s-images.idx3-ubyte' % 't10k'))):
            return ''

        sum = hashlib.md5()
        with open(os.path.join(self.path, '%s-labels.idx1-ubyte' % 'train'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, '%s-images.idx3-ubyte' % 'train'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, '%s-labels.idx1-ubyte' % 't10k'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, '%s-images.idx3-ubyte' % 't10k'), 'rb') as file:
            sum.update(file.read())

        return sum.hexdigest()

    def extract_files(self) -> list:
        return [
            os.path.join(self.path, '%s-labels.idx1-ubyte' % 'train'),
            os.path.join(self.path, '%s-images.idx3-ubyte' % 'train'),
            os.path.join(self.path, '%s-labels.idx1-ubyte' % 't10k'),
            os.path.join(self.path, '%s-images.idx3-ubyte' % 't10k')
        ]

    def estimate_size(self) -> int:
        return 62914560 #60MB

    def __load_core(self, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(self.path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(self.path, '%s-images.idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def load(self):
        train_x, train_y = self.__load_core(kind='train')
        test_x, test_y = self.__load_core(kind='t10k')

        return train_x, train_y, test_x, test_y
