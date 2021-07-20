import time

import network
import rpc


class Hello(rpc.AbsSimpleExecutor):

    def run(self, com: rpc.communication.Communication) -> object:
        for i in range(101):
            time.sleep(0.1)
            # Executor可以这样报告状态
            com.report_progress(i)

        return "Hello World"


if __name__ == '__main__':
    # 分配运行节点
    nodes = network.NodeAssignment()
    nodes.add(1, "127.0.0.1")

    # 创建执行环境
    with network.Request().request(nodes) as com:
        master = rpc.Coordinator(com)
        master.submit_group(Hello)
        res, err = master.join()
        # 打印结果
        print(res)
