import time

import network
import executor


class Hello(executor.AbsSimpleExecutor):

    def run(self, com: executor.communication.Communication) -> object:
        for i in range(101):
            time.sleep(0.1)
            com.report_progress(i)

        return "Hello World"


if __name__ == '__main__':
    nodes = network.NodeAssignment()
    nodes.add(1, "127.0.0.1")

    with network.Request().request(nodes) as com:
        master = executor.Coordinator(com)
        master.submit_group(Hello)
        res, err = master.join()
        print(res)
