from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interfaces import ITransfer


class PSGDOptimizer(IOptimizer):

    def __init__(self, gradient_descent:IGradientDescent, transfer:ITransfer, block_mgr:IPSGDBlockMgr):
        pass

    def optimize(self, *variables:ITrainable):
        pass

    def set_batch_size(self, batch_size: int):
        pass
