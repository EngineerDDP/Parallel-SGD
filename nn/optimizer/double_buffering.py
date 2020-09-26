from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from psgd.interface import ITransfer


class DoubleBufferingOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr
        self.__optimizer = gradient_descent
        self.__delayed_update_mark = False
        self.__batch_size = 1

    def optimize(self, variable: ITrainable):
        """
            Double buffering.
            Do receive before update
        :param variable: variable object.
        :return: None
        """
        # get last update result.
        if self.__delayed_update_mark:
            delta = self.__transfer.get_weights(variable.id, batch_no=self.__block_mgr.batch_id - 1)
            variable.set_value(variable.get_value() - delta)

        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)

        delta = self.__optimizer.delta(grad / self.__batch_size)
        self.__transfer.put_weights(delta, variable.id, self.__block_mgr.batch_id, self.__block_mgr.current_block_id)
        self.__delayed_update_mark = self.__block_mgr.end

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
