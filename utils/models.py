from enum import Enum

from profiles.settings import GlobalSettings


class Init(Enum):
    GlobalSettings = 'Req_GlobalSettings'
    Weights_And_Layers = 'Req_WeightsAndLayers'
    Codec_And_SGD_Type = 'Req_CodecAndSGD'
    Samples = 'Req_Samples'
    MISC = 'Req_OtherStuff'

class Reply:
    I_Need_Your_Working_Log = 'I need your working log'

    class global_setting_package:

        def __init__(self, settings: GlobalSettings):
            self.n = settings.nodes
            self.r = settings.redundancy
            self.b = settings.batch.batch_size
            self.ass = settings.block_assignment.__class__

        def restore(self):
            GlobalSettings.set_default(self.n, self.r, self.b, self.ass)

    class weights_and_layers_package:

        def __init__(self, layers: list):
            self.__storage = [(i.__class__, i.params()) for i in layers]

        def restore(self):
            return [i[0](*i[1]) for i in self.__storage]

    class codec_and_sgd_package:

        def __init__(self, codec, sgd_t):
            self.__codec = codec
            self.__sgd_t = sgd_t

        def restore(self):
            return self.__codec, self.__sgd_t

    class data_sample_package:

        def __init__(self, train_x, train_y, test_x, test_y):
            self.__train_x = train_x
            self.__train_y = train_y
            self.__test_x = test_x
            self.__test_y = test_y

        def restore(self):
            return self.__train_x, self.__train_y, self.__test_x, self.__test_y

    class misc_package:

        def __init__(self, epoch, loss_type, learn_rate, target_acc, w_types, op):
            self.epoch = epoch
            self.loss_type = loss_type
            self.learn_rate = learn_rate
            self.target_acc = target_acc
            self.w_types = w_types
            self.optimizer = op

class Ready_Type:

    def __init__(self):
        pass

class Binary_File_Package:

    def __init__(self, filename):

        self.filename = filename
        self.content = b''
        with open(filename, 'rb') as f:
            self.content = f.read()

    def restore(self):
        with open(self.filename, 'wb') as f:
            f.write(self.content)
