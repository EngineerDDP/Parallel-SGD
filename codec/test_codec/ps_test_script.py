if __name__ != '__main__':
    raise AssertionError('This module cannot be imported.')

# import test codec
from codec.essential import Block_Weight
from profiles.settings import GlobalSettings

# import np
import numpy as np

# import event logger
from utils.log import Logger

# import network agreements
from utils.constants import Parameter_Server


# 本模块为参数服务器编码测试模块，导入要测试的编码模块，分为服务器（主机）模块和客户端（工作）模块。
# 导入并定义 SLAVE_CODEC 变量为您的客户端测试编码类
# 导入并定义 MASTER_CODEC 变量为您的服务器测试编码类
# 运行本文件脚本，等待测试完成
# 如果运行过程中未发生错误，则证明编写的模块可以与本项目正常组合使用。
# 注意：本测试器只测试类的完整性，并不会测试编码过程的统计有效性，您需要使用数学证明来确认您的编码过程是完整有效的。

"""
    ---------------DEFINE HERE---------------
"""
from codec.sgq import SGQServer, SGQClient
# Type
SLAVE_CODEC = SGQClient
MASTER_CODEC = SGQServer
"""
    ---------------DEFINE HERE---------------
"""

# const parameters
SLAVE_IDS = [0,1,2,3]
MASTER_ID = -2
TEST_ROUNDS = 10
WEIGHTS_SHAPE = np.random.randint(3, 1024, size=2)
LAYER = 0
GlobalSettings.set_default(len(SLAVE_IDS), 1, 1, None)

# build codec
slave_codec = [SLAVE_CODEC(node_id=i) for i in SLAVE_IDS]
master_codec = MASTER_CODEC(node_id=MASTER_ID)

for i in range(TEST_ROUNDS):
    # starting consensus stage
    # set node
    node_id = 0
    for slave in slave_codec:
        # get random
        arr = np.random.random(size=WEIGHTS_SHAPE)
        # build BlockWeight
        blockweight = Block_Weight(LAYER, i, node_id, {node_id}, content=arr)
        # send consensus package
        for package in slave.update_blocks(blockweight):
            # check the package that will be sent to parameter server
            assert Parameter_Server in package.target()
            # reply each package
            for reply in master_codec.receive_blocks(package.content()):
                # check the package header
                assert node_id in reply.target()
                # receive each reply
                slave.receive_blocks(reply.content())
        arr_res = slave.get_result()
        # inc
        node_id += 1

    print("INFO: -----------Test complete {}/{} -----------".format(i, TEST_ROUNDS))

for slave in slave_codec:
    slave.dispose()

master_codec.dispose()

print("INFO: All test input was handled without exception.")
print("WARNING: The functionality of the codec cannot be tested here.\n"
      "WARNING: Use Mathematical analysis to make sure that your codec process didn't prevents SGD from properly convergence.")


