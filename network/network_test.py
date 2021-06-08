import threading
import unittest

import network


class TestCase(unittest.TestCase):
    TEST_ROUND = 10
    TEST_DATA = b'a' * 1024 * 1024 * 100  # 100 MB data

    def __init__(self, methodName: str):
        super().__init__(methodName=methodName)
        self.__count = 0

    def run_serve(self):
        serve = network.Serve()
        count = 0

        with serve.acquire() as com:
            com: network.ICommunication_Controller
            try:
                while not com.is_closed():
                    _, data = com.get_one(blocking=True, timeout=1)
                    if data is not None:
                        count += 1
            except ConnectionAbortedError:
                pass
        print("{} acquired, {} required".format(count, TestCase.TEST_ROUND))
        self.__count = count

    @staticmethod
    def run_request():
        request = network.Request()
        node = network.NodeAssignment()
        node.add(0, '127.0.0.1')

        com = request.request(node)
        com.establish_communication()
        for i in range(TestCase.TEST_ROUND):
            com.send_one(0, TestCase.TEST_DATA)
        com.close(force=False, timeout=100)

    def test_something(self):
        print(network.communications.get_repr())

        t_serve = threading.Thread(target=self.run_serve, name="Serve T")
        t_request = threading.Thread(target=self.run_request, name="Request T")

        t_serve.start()
        t_request.start()
        t_serve.join()
        t_request.join()

        self.assertEqual(self.__count, TestCase.TEST_ROUND)


if __name__ == '__main__':
    unittest.main()
