from codec.essential import Block_Weight
from profiles.settings import GlobalSettings

# import np
import numpy as np

# import event logger
from utils.log import Logger
from time import sleep


# 本模块为P2P训练网络编码测试模块，导入要测试的编码模块以测试您的模块与系统的兼容性。
# 导入并定义 SLAVE_CODEC 变量为您的客户端测试编码类
# 导入并定义 ASSIGNMENTS 指定您设置的数据集分配方案
# 运行本文件脚本，等待测试完成
# 如果运行过程中未发生错误，则证明编写的模块可以与本项目正常组合使用。
# 注意：本测试器只测试类的完整性，并不会测试编码过程的统计有效性，您需要使用数学证明来确认您的编码过程是完整有效的。

"""
    ---------------DEFINE HERE---------------
"""
# import test codec
from codec.tutorial_codec import myComCtrl
from profiles.blockassignment.duplicate import DuplicateAssignment
# Type
SLAVE_CODEC = myComCtrl
ASSIGNMENTS = DuplicateAssignment
"""
    ---------------DEFINE HERE---------------
"""

# const parameters
SLAVE_CNT = 4
REDUNDANCY = 1
TEST_ROUNDS = 10
WEIGHTS_SHAPE = np.random.randint(3, 1024, size=2)
LAYER = 0
BATCHSIZE = 64
SYNCWAITTIMEOUT = 1000 #ms

# setup global parameters
GlobalSettings.set_default(SLAVE_CNT, REDUNDANCY, BATCHSIZE, ASSIGNMENTS)

# default setting
Default = GlobalSettings.get_default()

# build codec
slave_codec = [SLAVE_CODEC(node_id=i) for i in range(SLAVE_CNT)]

for i in range(TEST_ROUNDS):
    # starting consensus stage
    node_id = 0
    for slave in slave_codec:
        # build each block
        for block_id in Default.block_assignment.node_2_block[node_id]:
            # get random
            arr = np.random.random(size=WEIGHTS_SHAPE)
            # build blockweights
            blockweight = Block_Weight(LAYER, i, block_id, Default.block_assignment.block_2_node[block_id], arr)
            # send consensus package
            for package in slave.update_blocks(blockweight):
                # get proper receiver
                for tgt in package.target():
                    assert tgt in range(SLAVE_CNT)
                    recv = slave_codec[tgt]
                    # recv pkg
                    recv.receive_blocks(package.content())
                    print("INFO: ----------- Node:{} Transmitting to {} successful -----------".format(node_id, tgt))

        node_id += 1

    node_id = 0
    for slave in slave_codec:
        # wait until done
        time_out = 0
        retried = False
        while not slave.is_done():
            sleep(0.001)
            time_out += 1
            if time_out > SYNCWAITTIMEOUT and not retried:
                print("WARNING: Timeout occurred while get the result from worker {}.".format(node_id))
                do_retry = slave.do_something_to_save_yourself()
                if do_retry is None:
                    raise TimeoutError('Timeout occurred and retry mechanism is not available.')
                for package in do_retry:
                    # get proper receiver
                    for tgt in package.target():
                        assert tgt in range(SLAVE_CNT)
                        recv = slave_codec[tgt]
                        # recv pkg
                        recv.receive_blocks(package.content())
                        print("INFO: ----------- Node:{} Backup to {} successful -----------".format(node_id, tgt))
                time_out = 0
                retried = True
            elif time_out > SYNCWAITTIMEOUT and retried:
                raise TimeoutError('Decode timeout.')

        arr_res = slave.get_result()
        print("INFO: ----------- Node:{} Decode successful -----------".format(node_id))

        node_id += 1

    print("INFO: -----------Test complete {}/{} -----------".format(i, TEST_ROUNDS))

for slave in slave_codec:
    slave.dispose()


print("INFO: All test input_ref was handled without exception.")
print("WARNING: The functionality of the codec cannot be tested here.\n"
      "WARNING: Use Mathematical analysis to make sure that your codec process didn't prevents SGD from properly convergence.")

