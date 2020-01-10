import pickle


train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_batch = ['test_batch']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(path='./dataset/cirfar_data/', kind='train'):

    if kind == 'train':
        for train_file in train_batch:
            dict = unpickle(path+train_file)

