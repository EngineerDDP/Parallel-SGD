import numpy as np

from nn import IOptimizer, ITrainable
from nn.data.block_data_feeder import IPSGDBlockMgr
from nn.gradient_descent.interface import IGradientDescent
from parallel_sgd.batch_sync.interface import ITransfer


class GradientAveragingOptimizer(IOptimizer):
    """
        P-SGD Optimizer
        Interact with transfer.
    """

    def __init__(self, gradient_descent: IGradientDescent, transfer: ITransfer, block_mgr: IPSGDBlockMgr):
        self.__transfer = transfer
        self.__block_mgr = block_mgr
        self.__optimizer = gradient_descent
        self.__batch_size = 1
        self.__initial_value: [np.ndarray] = None

    def optimize(self, variable: ITrainable):
        """
            1st order gradient based optimize algorithm.
            {arg min}_{x}{F(x)}
        :param variable: variable object.
        :return: None
        """
        if self.__initial_value is None:
            self.__initial_value = variable.get_value()

        grad = variable.get_gradient()
        if variable.get_shape() != grad.shape:
            grad = grad.sum(axis=0)

        self.__transfer.put_weights(self.__optimizer.delta(grad / self.__batch_size), variable.id,
                                    self.__block_mgr.batch_id,
                                    self.__block_mgr.current_block_id)
        if self.__block_mgr.end:
            new_parameter = self.__transfer.get_weights(variable.id, batch_no=self.__block_mgr.batch_id)
            variable.set_value(new_parameter + self.__initial_value)

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
