import network

if __name__ == '__main__':
    serve = network.Serve()
    with serve.acquire() as com:
        com: network.ICommunication_Controller
        for i in range(64):
            com.get_one()
            print(i)
        com.close()

    print("Exit")