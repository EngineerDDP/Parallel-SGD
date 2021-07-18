import multiprocessing
import shutil
import time
import unittest

import roles
from executor.abstract import AbsSimpleExecutor
from executor.communication import Communication
from network import NodeAssignment
from network import Request


class DeadLoop(AbsSimpleExecutor):

    def __init__(self, node_id: int, working_group: set, initializer_id):
        super().__init__(node_id, working_group, initializer_id)

    def run(self, com: Communication) -> None:
        while True:
            com.get_one(blocking=True, timeout=None)


class TestExecutor(unittest.TestCase):

    def test_shutdown(self):
        worker = multiprocessing.Process(target=roles.Cohort().slave_forever)
        worker.start()

        nodes = NodeAssignment()
        nodes.add(101, '127.0.0.1')

        net = Request()
        with net.request(nodes) as req:
            master = roles.Coordinator(req)
            master.submit_group(DeadLoop, package_size=18000)
            time.sleep(5)
            master.stop_executing()
            master.join()
            del master

        time.sleep(1)
        worker.kill()
        time.sleep(2)
        shutil.rmtree("./Node-101-Retrieve")
        shutil.rmtree("./tmp_log")


if __name__ == '__main__':
    unittest.main()
