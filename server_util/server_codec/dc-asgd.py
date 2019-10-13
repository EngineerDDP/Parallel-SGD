# import numpy as np
#
# from codec.pacodec import PACodec
#
# from settings import GlobalSettings
#
#
# class DC_ASGD_a(PACodec):
#     """
#         Implemented according to paper:
#         Shuxin Zheng, Qi Meng, Taifeng Wang, et al. Asynchronous Stochastic Gradient Descent with Delay Compensation.
#         International Conference on Machine Learning (ICML), Sydney, Australia. 2017.
#     """
#
#     def __init__(self, weights_init, layer_id, learn_rate):
#
#         # init weights
#         self.Weights_init = weights_init
#
#         # other parameters
#         self.Layer_ID = layer_id
#         self.Learn_Rate = learn_rate
#         self.Bak_Weights_Node = {}
#         self.Lambda_T = 2
#         self.Mean_Square = 0
#         self.Mean_Square_Epsilon = 1e-7
#         self.Mean_Square_M = 0.95
#
#         # init w_bak
#         for key in GlobalSettings.getDefault().Nodes:
#             self.Bak_Weights_Node[key] = self.Weights_init
#
#     def acquire_weights(self, tag):
#
#         self.Bak_Weights_Node[tag.Node_No] = self.Weights_init
#         return self.Weights_init
#
#     def update_blocks(self, content, tag):
#
#         delay = np.multiply(np.multiply(content, content), self.Weights_init - self.Bak_Weights_Node[tag.Node_No])
#         self.Weights_init = self.Weights_init - self.Learn_Rate * (content + self.Lambda_T * delay)
#         self.Mean_Square = self.Mean_Square_M * self.Mean_Square + (1 - self.Mean_Square_M) * np.multiply(content, content)
#         self.Lambda_T = self.Lambda_T / np.sqrt(self.Mean_Square + self.Mean_Square_Epsilon)