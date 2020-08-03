from network.communications import Communication_Controller
from network.starnet_com_process import start_star_net, StarNetwork_Initialization_Package
import random


if __name__ == '__main__':
    check_code = random.random()
    package = StarNetwork_Initialization_Package()
    worker_list = [
        ('0', check_code, '10.199.196.236'),
        #(1, check_code, '192.168.1.112'),
        #('2', check_code, '192.168.1.122')
    ]
    for worker in worker_list:
        package.put(*worker)

    con = start_star_net(package)
    com = Communication_Controller(con)
    com.establish_communication()

    # while True:
    #     data = com.get_one()
    #     print(data)
