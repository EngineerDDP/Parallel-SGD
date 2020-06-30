import time
import pandas as pd

from threading import Thread

from profiles.settings import GlobalSettings

from network.communications import CommunicationController
from network.agreements import General, Initialize, DefaultNodes

from neuralnetworks.metrics import CategoricalAccuracy
from neuralnetworks.optimizer import ParallelSGDOptimizer, GradientDecentOptimizer_v2, ParallelSGDWithPSOptimizer, DelayedPSGDOptimizer
from neuralnetworks.model import SequentialModel_v2

from psgd.transfer import NTransfer

from codec.tag import Tag

from dataset.mnist_input import load_mnist

from log import Logger

import sys


if __name__ == '__main__':

    # 设置远端连接地址和端口
    CommunicationController.static_server_address = "127.0.0.1"
    CommunicationController.static_server_port = 55555

    # 设置好连接控制器
    con = CommunicationController()

    # 建立连接
    con.establish_communication()

    print();