import time

import network
import numpy as np


if __name__ == '__main__':
    request = network.Request()
    node = network.NodeAssignment()
    node.add(0, '127.0.0.1')

    with request.request(node) as com:
        while not com.is_closed():
            com: network.ICommunication_Controller
            for i in range(100):
                com.send_one(0, np.random.uniform(size=[100000]))
                print(i)
            time.sleep(1)
