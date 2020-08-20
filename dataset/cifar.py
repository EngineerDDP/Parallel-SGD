import pickle
import numpy as np


train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_batch = ['test_batch']


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(path='./dataset/cirfar_data/', kind='train'):

    label = []
    data = []

    if kind == 'train':
        for train_file in train_batch:
            dict = unpickle(path+train_file)
            label.append(dict[b'labels'])
            data.append(dict[b'data'])

    elif kind == 't10k':
        for test_file in test_batch:
            dict = unpickle(path+test_file)
            label.append(dict[b'labels'])
            data.append(dict[b'data'])

    else:
        raise AssertionError('kind parameter can only be either train or t10k.')

    label = np.concatenate(label, axis=0)
    data = np.concatenate(data, axis=0)

    return data, label


def load():
    train_x, train_y = load_data(kind='train')
    test_x, test_y = load_data(kind='t10k')

    return train_x, train_y, test_x, test_y