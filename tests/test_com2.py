import network
import time
import models


class F:

    def do(self, a, b):
        print("Hello")
        return a + b


if __name__ == '__main__':
    request = network.Request()
    node = network.NodeAssignment()
    node.add(0, '127.0.0.1')

    with request.request(node) as com:
        com.send_one(0, models.ClassSerializer(F))
        time.sleep(2)

        while not com.is_closed():
            s = input("->")
            com.send_one(0, s)
            print(":{}".format(s))
            print(com.get_one()[1])
