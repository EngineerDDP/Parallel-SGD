from network.communications import *
from network.starnet_com_process import *


if __name__ == '__main__':

    reg = Worker_Communication_Constructor('0.0.0.0', STAR_NET_WORKING_PORTS, Worker_Register()).buildCom()
    com = Communication_Controller(Communication_Process(reg))
    com.establish_communication()

    for i in range(1):
        com.send_one([-1], {'abc': 'def'})
        print("send one")
