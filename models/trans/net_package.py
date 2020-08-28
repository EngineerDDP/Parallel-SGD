from executor.interfaces import IExecutor
from dataset.interfaces import AbsDataset, IDataset
from dataset.transforms.__init__ import ITransformer
from models.local import IServerModel
from models.trans import IReplyPackage, ClassSerializer, Binary_File_Package
from profiles import Settings


class SubmitJob(IReplyPackage):

    def __init__(self, nodes:set, group_offset:int, ps:bool, eta_waiting_time:int, exe:type):
        self.__nodes = nodes
        self.__ps = ps
        self.__eta_wait = eta_waiting_time
        self.__cls  = ClassSerializer(exe)
        self.__offset = group_offset

    def restore(self) -> None:
        self.__cls : type = self.__cls.restore()

    @property
    def group_offset(self):
        return self.__offset

    @property
    def executioner(self):
        return self.__cls

    @property
    def work_group(self) -> set:
        return self.__nodes

    @property
    def am_i_ps(self):
        return self.__ps

    @property
    def waiting_time(self):
        return self.__eta_wait


class global_setting_package(IReplyPackage, Settings):

    def __init__(self, settings:Settings):
        self.n = settings.node_count
        self.r = settings.redundancy
        self.b = settings.batch.batch_size
        self.ass = settings.block_assignment
        self.__sets = None

    def restore(self):
        self.__sets = Settings(self.n, self.r, self.b, self.ass)

    @property
    def redundancy(self) -> int:
        return self.__sets.redundancy

    @property
    def node_count(self) -> int:
        return self.__sets.node_count

    @property
    def block_assignment(self):
        return self.__sets.block_assignment

    @property
    def batch(self):
        return self.__sets.batch

    @property
    def nodes(self) -> set:
        return self.__sets.nodes


class essentials(IReplyPackage, IServerModel):
    """
        Decorator for IServerModel instance.
    """

    def __init__(self, model:IServerModel):
        # non RPC object
        self.__weights_types = model.weights_types
        self.__loss_type = model.loss_type
        self.__metric = model.metric
        self.__target_acc = model.target_acc
        self.__psgd_type = model.psgd_type
        self.__psgd_server_type = model.psgd_server_type
        self.__optimizer_type = model.optimizer_type
        self.__epochs = model.epoches
        self.__learn_rate = model.learn_rate
        # serialized object
        self.__storage = [(i.__class__, i.param(), i.Variables, i.Act.__class__) for i in model.get_nn()]
        # RPC class
        self.__codec_ctrl = [ClassSerializer(cc) for cc in model.codec_ctrl]
        # None PS
        if model.psgd_server_codec is not None:
            self.__psgd_server_codec = ClassSerializer(model.psgd_server_codec)
        else:
            self.__psgd_server_codec = None
        # decorated object
        self.__get_nn = None

    def restore(self):
        self.__codec_ctrl = [sc.restore() for sc in self.__codec_ctrl]
        self.__get_nn = [i[0](*i[1], *i[2], i[3]()) for i in self.__storage]
        if isinstance(self.__psgd_server_codec, IReplyPackage):
            self.__psgd_server_codec = self.__psgd_server_codec.restore()

    @property
    def weights_types(self):
        return self.__weights_types

    def get_nn(self):
        return self.__get_nn

    @property
    def loss_type(self):
        return self.__loss_type

    @property
    def codec_ctrl(self):
        return self.__codec_ctrl

    @property
    def metric(self):
        return self.__metric

    @property
    def target_acc(self):
        return self.__target_acc

    @property
    def psgd_type(self):
        return self.__psgd_type

    @property
    def psgd_server_codec(self):
        return self.__psgd_server_codec

    @property
    def psgd_server_type(self):
        return self.__psgd_server_type

    @property
    def optimizer_type(self):
        return self.__optimizer_type

    @property
    def epoches(self):
        return self.__epochs

    @property
    def learn_rate(self):
        return self.__learn_rate


class data_package(IReplyPackage, IDataset):

    def __init__(self, data_cls:AbsDataset, transform:ITransformer):
        self.__sum = data_cls.check_sum()
        self.__cls = ClassSerializer(data_cls.__class__)
        self.__decorated_class : AbsDataset = None
        self.__transformer = transform

    def restore(self):
        self.__decorated_class = self.__cls.restore()(self.__sum)

    def load(self) -> tuple:
        return self.__transformer(*self.__decorated_class.load())

    def check(self):
        return self.__decorated_class.check()

    def __repr__(self):
        return self.__decorated_class.__repr__()

class data_content(data_package):

    def __init__(self, data_cls:IDataset, transform:ITransformer):
        super().__init__(data_cls, transform)
        self.__contents = [Binary_File_Package(filename) for filename in data_cls.extract_files()]

    def restore(self) -> None:
        super().restore()
        for b_file in self.__contents:
            b_file.restore()
