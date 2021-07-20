from typing import Dict

import numpy as np
import os
import hashlib

from nn.dataset.interfaces import AbsDataset


class CIFAR(AbsDataset):

    def __init__(self, check_sum=None):
        self.path = './.data/cifar_data/'
        super().__init__(check_sum)

    def __repr__(self):
        return '<CIFAR-10 classification dataset.>'

    def load(self) -> tuple:
        data: Dict[str, np.ndarray] = np.load(self.path + "cifar10", allow_pickle=True)[()]
        return data["x_train"], data["y_train"].reshape(-1), data["x_test"], data["y_test"].reshape(-1)

    def check_sum(self) -> str:
        if not os.path.exists(self.path + "cifar10"):
            return ''
        sum = hashlib.md5()
        with open(self.path + "cifar10", 'rb') as f:
            sum.update(f.read())
        return sum.hexdigest()

    def extract_files(self) -> list:
        files = [self.path + "cifar10"]
        return files

    def estimate_size(self) -> int:
        return 184380559  #175MB
