import numpy as np

def make_onehot(label):
    return np.eye(10)[label]


def make_image_scale(sample):
    return sample / 255
