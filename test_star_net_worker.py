from network.communications import Worker_Communication_Constructor, Communication_Controller
from network.starnet_com_process import Worker_Register, Communication_Process, STAR_NET_WORKING_PORTS
from time import sleep


if __name__ == '__main__':

    reg = Worker_Communication_Constructor('0.0.0.0', STAR_NET_WORKING_PORTS, Worker_Register()).buildCom()
    con = Communication_Process(reg)
    com = Communication_Controller(con)
    com.establish_communication()

    for i in range(1):
        com.send_one([-1], {'abc': 'def'})
        print("send one")

    sleep(2)
    com.close()