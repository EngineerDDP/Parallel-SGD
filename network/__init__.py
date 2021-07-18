# for import usage
import time

from network.interfaces import ICommunicationController, NodeAssignment

from network.communications import WorkerCommunicationConstructor as wcc
from network.communications import CommunicationController as cc


class Request:

    def __init__(self, net_type='fcnet'):
        """
            Only fcnet is available right now.
        :param net_type: 'fcnet'
        """
        net_type = net_type.lower()
        if net_type == 'fcnet':
            from network.starnet_com_process import Promoter as FCNet
            self.__method = FCNet()
        else:
            raise AssertionError('Cannot find network type that matches {}.'.format(net_type))

    def request(self, nodes:NodeAssignment) -> ICommunicationController:
        """
            Startup a network and do something
        """
        com_proc = self.__method(nodes)
        com_ctrl = cc(com_proc)
        time.sleep(2)
        return com_ctrl


class Serve:

    def __init__(self, net_type:str='fcnet'):
        """
            Only fcnet is available right now.
        :param net_type: 'fcnet'
        """
        net_type = net_type.lower()
        if net_type == 'fcnet':
            from network.starnet_com_process import CommunicationProcess, NodeRegister
            self.__constructor = wcc(NodeRegister())
            self.__proc_cls = CommunicationProcess
        else:
            raise AssertionError('Cannot find network type that matches {}.'.format(net_type))

    def acquire(self) -> ICommunicationController:
        """
            Receive a connection from promoter.
        :return: full functional ICommunication_Controller instance
        """
        register = self.__constructor.build_communication()
        self.__constructor.close()
        com_proc = self.__proc_cls(register)
        com_ctrl = cc(com_proc)
        return com_ctrl

    def close(self):
        self.__constructor.close()