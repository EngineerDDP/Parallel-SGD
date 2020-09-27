from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interface import ITransfer


class PSGDOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr
        self.__optimizer = gradient_descent
        self.__batch_size = 1

    def optimize(self, variable: ITrainable):
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.mean(axis=0)
        delta = self.__optimizer.delta(grad)
        self.__transfer.put_weights(delta, variable.id, self.__block_mgr.batch_id, self.__block_mgr.current_block_id)
        if self.__block_mgr.end:
            delta = self.__transfer.get_weights(variable.id, batch_no=self.__block_mgr.batch_id)
            variable.set_value(variable.get_value() - delta)

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
