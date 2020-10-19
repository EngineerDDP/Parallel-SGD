import network

if __name__ == '__main__':
    serve = network.Serve()
    with serve.acquire() as com:
        _, cls = com.get_one()
        cls.restore()
        obj = cls()
        print(obj.do(1, 1))

        while not com.is_closed():
            s = input("->")
            com.send_one(-1, s)
            print(":{}".format(s))
            print(com.get_one()[1])
