import pickle
import numpy as np
import os
import hashlib

from dataset.interfaces import AbsDataset


class CIFAR(AbsDataset):

    train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = ['test_batch']

    def __init__(self, check_sum=None):
        self.path = './dataset/cirfar_data/'
        super().__init__(check_sum)

    def __repr__(self):
        return '<CIFAR-10 classification dataset.>'

    @staticmethod
    def __unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __load_data(self, kind='train'):

        label = []
        data = []

        if kind == 'train':
            for train_file in CIFAR.train_batch:
                dict = CIFAR.__unpickle(self.path + train_file)
                label.append(dict[b'labels'])
                data.append(dict[b'data'])

        elif kind == 't10k':
            for test_file in CIFAR.test_batch:
                dict = CIFAR.__unpickle(self.path + test_file)
                label.append(dict[b'labels'])
                data.append(dict[b'data'])

        else:
            raise AssertionError('kind parameter can only be either train or t10k.')

        label = np.concatenate(label, axis=0)
        data = np.concatenate(data, axis=0)

        return data, label

    def load(self) -> tuple:
        train_x, train_y = self.__load_data(kind='train')
        test_x, test_y = self.__load_data(kind='t10k')

        return train_x, train_y, test_x, test_y

    def check_sum(self) -> str:
        for train_file in CIFAR.train_batch + CIFAR.test_batch:
            if not os.path.exists(self.path + train_file):
                return ''

        sum = hashlib.md5()
        for train_file in CIFAR.train_batch + CIFAR.test_batch:
            with open(self.path + train_file, 'rb') as f:
                sum.update(f.read())

        return sum.hexdigest()

    def extract_files(self) -> list:
        files = [self.path + file for file in CIFAR.test_batch + CIFAR.train_batch]
        return files

    def estimate_size(self) -> int:
        return 209715200 #200MB