# import test codec
from codec.essential import BlockWeight
from codec.interfaces import netEncapsulation

# import np
import numpy as np

# import event logger
from codec import GlobalSettings
from profiles import Settings
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
from codec.quantization import Quantization2BitPSCodec, FPWParaServer
from codec.pdd import JianShang
# Type
SLAVE_CODEC = Quantization2BitPSCodec
MASTER_CODEC = FPWParaServer
"""
    ---------------DEFINE HERE---------------
"""

# const parameters
SLAVE_IDS = [0,1,2,3]
MASTER_ID = -2
TEST_ROUNDS = 1024
WEIGHTS_SHAPE = np.random.randint(3, 1024, size=2)
LAYER = 0
GlobalSettings.deprecated_default_settings = Settings(len(SLAVE_IDS), 1, 1, None)

# build codec
codecs = {}
slave_codec = []
for id in SLAVE_IDS:
    codecs[id] = SLAVE_CODEC(node_id=id)
    slave_codec.append(codecs[id])

codecs[MASTER_ID] = MASTER_CODEC(node_id=MASTER_ID)

for i in range(TEST_ROUNDS):
    # starting consensus stage
    # set node
    node_id = 0
    for slave in slave_codec:
        # get random
        arr = np.random.random(size=WEIGHTS_SHAPE)
        # build BlockWeight
        blockweight = BlockWeight(LAYER, i, node_id, {node_id}, content=arr)
        # send consensus package
        pkg = slave.update_blocks(blockweight)
        if isinstance(pkg, netEncapsulation):
            pkg = [pkg]
        if pkg is None:
            pkg = []
        for package in pkg:
            # check the package that will be sent to parameter server
            for target in package.target():
                # send each package
                replies = codecs[target].receive_blocks(package.content())
                if isinstance(replies, netEncapsulation):
                    replies = [replies]
                if replies is None:
                    replies = []
                # reply each package
                for reply in replies:
                    # check the package header
                    for node_id in reply.target():
                        # receive each reply
                        codecs[node_id].receive_blocks(reply.content())
        # inc
        node_id += 1
    for slave in slave_codec:
        arr_res = slave.get_result()


    print("INFO: -----------Test complete {}/{} -----------".format(i, TEST_ROUNDS))

for slave in slave_codec:
    slave.dispose()

codecs[MASTER_ID].dispose()


print("INFO: All test input_ref was handled without exception.")
print("WARNING: The functionality of the codec cannot be tested here.\n"
      "WARNING: Use Mathematical analysis to make sure that your codec process didn't prevents SGD from properly convergence.")


