from abc import abstractmethod
from enum import Enum
from typing import List, Type, Union

from numpy import ndarray

from codec.interfaces import Codec
from dataset.interfaces import AbsDataset, IDataset
from dataset.transforms.interface import ITransformer
from models import IReplyPackage, ClassSerializer, IRequestPackage, BinaryFilePackage
from network import ICommunication_Controller
from nn import IModel, IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from nn.optimizer import IOptimize
from profiles import ISetting
from profiles.interface import IBatchIter
from psgd.interface import ITransfer
from psgd.sync import IParallelSGD
from psgd.transfer import NTransfer
from utils.log import IPrinter


class IPSGDOptimize(IOptimize):

    @abstractmethod
    def assemble(self, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        pass


class Req(Enum):
    Setting = "Setting"
    Model = "Model"
    Optimizer = "Optimizer"
    Transfer = "Transfer"
    Transfer_PS = "TransferPS"
    Data_Package = "Data"
    Data_Content = "Samples"
    Other_Stuff = "MISC"


class Requests(IRequestPackage):

    def __init__(self, req: Req):
        self.__req = req

    def content(self) -> object:
        return self.__req

    def __repr__(self):
        return "<P-SGD Requests ({})>".format(self.__req.value)

    def __str__(self):
        return self.__repr__()


class net_setting(IReplyPackage):

    def __init__(self, ba_type: type, *params):
        self.__type: type = ba_type
        self.__params = params
        self.__core: [ISetting] = None

    def restore(self) -> None:
        self.__core = self.__type(*self.__params)

    def setting(self):
        return self.__core


class net_model(IReplyPackage):
    """
        Decorator for IServerModel instance.
    """

    def __init__(self, model: IModel, batch_iter: IBatchIter):
        self.model = model
        self.batch_iter = batch_iter

    def restore(self):
        pass

    def __repr__(self):
        return "<Net package (Model)>"


class net_optimizer(IReplyPackage, IPSGDOptimize):

    def __init__(self, optimizer_type: Type[IOptimizer], gradient_method: Type[IGradientDescent], op_params=tuple()):
        self.__op_params = op_params
        self.__ops: List[IOptimizer] = []

        self.__op: Union[ClassSerializer, Type[IOptimizer]] = ClassSerializer(optimizer_type)
        self.__gd: Union[ClassSerializer, Type[IGradientDescent]] = ClassSerializer(gradient_method)
        self.__trans: [ITransfer] = None
        self.__b_mgr: [IPSGDBlockMgr] = None

    def restore(self) -> None:
        self.__op = self.__op.restore()
        self.__gd = self.__gd.restore()

    def assemble(self, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__trans = transfer
        self.__b_mgr = block_mgr

    def optimize(self, *variables: ITrainable):
        for var in variables:
            self.__ops.append(self.__op(
                gradient_descent=self.__gd(*self.__op_params),
                transfer=self.__trans,
                block_mgr=self.__b_mgr))
            var.attach_optimizer(self.__ops[-1])

    def set_batch_size(self, batch_size: int):
        for var in self.__ops:
            var.set_batch_size(batch_size)

    def __repr__(self):
        return "<Net package (Optimize)>"


class net_transfer(ITransfer, IReplyPackage):

    def __init__(self, var_ids, sgd_type: Type[IParallelSGD], codec_type: Type[Codec]):
        self.__vars: List[int] = var_ids
        self.__sgd_type: Type[IParallelSGD] = sgd_type
        self.__codec_type: Union[Type[Codec], ClassSerializer] = ClassSerializer(codec_type)
        self.__trans: [ITransfer] = None

    def restore(self) -> None:
        self.__codec_type = self.__codec_type.restore()

    def start_transfer(self, com: ICommunication_Controller, group_offset: int, printer: IPrinter) -> None:
        weight_ctrl = {var_id: self.__sgd_type(self.__codec_type(com.Node_Id)) for var_id in self.__vars}
        self.__trans = NTransfer(weight_ctrl)
        # redirect function
        self.put_weights = self.__trans.put_weights
        self.get_weights = self.__trans.get_weights
        self.__trans.start_transfer(com, group_offset, printer)

    def put_weights(self, content: ndarray, var_id: int, batch_no: int, block_id: int) -> None:
        pass

    def get_weights(self, var_id: int, batch_no: int) -> ndarray:
        pass

    def __repr__(self):
        return "<Net package (Transfer)>"


class data_package(IReplyPackage, IDataset):

    def __init__(self, data_cls: AbsDataset, transform: ITransformer):
        self.__sum = data_cls.check_sum()
        assert self.__sum != '', "Local dataset is corrupted."
        self.__cls = ClassSerializer(data_cls.__class__)
        self.__decorated_class: [AbsDataset] = None
        self.__transformer = transform

    def restore(self):
        self.__decorated_class = self.__cls.restore()(check_sum=self.__sum)

    def load(self) -> tuple:
        return self.__transformer(*self.__decorated_class.load())

    def check(self):
        return self.__decorated_class.check()

    def __repr__(self):
        if self.__decorated_class is not None:
            return self.__decorated_class.__repr__()
        else:
            return "<Net package (Dataset)>"


class data_content(data_package):

    def __init__(self, data_cls: AbsDataset, transform: ITransformer):
        super().__init__(data_cls, transform)
        self.__contents = [BinaryFilePackage(filename) for filename in data_cls.extract_files()]

    def restore(self) -> None:
        super().restore()
        for b_file in self.__contents:
            b_file.restore()


class misc_package(IReplyPackage):

    def __init__(self, mission_title, epoch, target_acc):
        self.mission_title = mission_title
        self.epoch = epoch
        self.target_acc = target_acc

    def restore(self) -> None:
        pass

    def __repr__(self):
        return "<Net package (Other stuff)>"
