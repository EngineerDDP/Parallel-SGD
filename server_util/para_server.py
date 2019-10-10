from server_util.init_model import ServerUtil

from psgd.interfaces import IParallelSGD
from codec.pacodec import PAServerCompack, PAClientComPack

from server_util.codec.interfaces import PACodec
from network.agreements import DefaultNodes


class ParameterServer(IParallelSGD):

    def __init__(self,node_id, layer_id, codec):

        super().__init__(node_id, layer_id, codec)


    def init_startup_setting(self, params=None):
        pass

    def release_memory(self):
        pass



class ParameterServer:

    def __init__(self, ctrl=PACodec):

        self.Layer_Controller = {}
        self.Controller_Builder = ctrl

        lr = ServerUtil.learn_rate()
        layer_no = 0

        for i in ServerUtil.getWeightsInit():
            self.Layer_Controller[layer_no] = {'w': self.Controller_Builder(i[1], layer_no, lr),
                                               'b': self.Controller_Builder(i[2], layer_no, lr)}

    def putWeights(self, obj):

        compack = PAClientComPack.decompose_compack(obj)
        ctrl = self.Layer_Controller[compack.Layer_ID]

    def acquireWeights(self):

        pass