import numpy as np

from dataset.interfaces import AbsDataset


class NoiseSimulation:

    def __init__(self, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.NScale = normal_scale
        self.BScale = bin_scale
        self.BRate = bin_rate
        self.Oneside = oneside

    def predict(self, x):
        # Gaussian noise
        n1 = np.random.normal(0.0, self.NScale, size=x.shape)
        # select points
        b1 = np.random.binomial(1, self.BRate, size=x.shape)
        if not self.Oneside:
            s1 = b1[np.where(b1 == 1)].shape
            # select side
            s1 = self.BScale * np.random.binomial(1, 0.5, size=s1)
            s1[np.where(s1 == 0)] = -1 * self.BScale
            # write back
            b1[np.where(b1 == 1)] = s1
        else:
            b1[np.where(b1 == 1)] = self.BScale

        return x + n1 + b1


class LinearSimulation:

    def __init__(self, w, b=0.0, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = w * x + b
        """

        self.W = w
        self.B = b

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):
        """
            Create samples with noise
        """

        return self.Noise.predict(np.dot(x, self.W.T) + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return np.dot(x, self.W) + self.B


class SinSimulation:

    def __init__(self, a=2.0, b=0.0, w=2*np.pi, normal_scale=1.0, bin_scale=1.0, bin_rate=0.1, oneside=True):
        """
            build simulation model base line
            model creates samples like y = sin(x * 2*pi/w) + b
        """

        self.B = b
        self.W = w
        self.A = a

        self.Noise = NoiseSimulation(normal_scale, bin_scale, bin_rate, oneside)

    def predict(self, x):

        return self.Noise.predict(np.sin(x * 2 * np.pi / self.W) + self.B)

    def baseline(self, x):
        """
            Create base line
        """
        return self.A * np.sin(x * 2 * np.pi / self.W) + self.B


class SimLin(AbsDataset):

    def __init__(self, check_sum:str=None, input_shape:int=1024, output_shape:int=1):
        """
            Interesting way to identify input_ref and output_ref scale.
            Use checksum string to storage input_ref and output_ref data.
        :param check_sum: checksum string, nothing need to be check, this string is used to transfer
                            shape of samples.
        :param input_shape: input_ref shape, int
        :param output_shape: output_ref shape, int
        """
        if check_sum is None:
            check_sum = '{},{}'.format(input_shape, output_shape)
        tmp = check_sum.split(',')
        self.__input_shape, self.__output_shape = int(tmp[0]), int(tmp[1])
        super().__init__(check_sum)

    def load(self):
        return SimLin.__load(self.__input_shape, self.__output_shape)

    def check(self):
        return True

    def check_sum(self) -> str:
        return '{},{}'.format(self.__input_shape, self.__output_shape)

    def extract_files(self) -> list:
        return []

    def estimate_size(self) -> int:
        return self.__input_shape * 60000

    @staticmethod
    def __load(len_x:int=1024, len_y:int=1):

        x = np.random.uniform(0, 10, size=[60000, len_x])
        w = np.random.uniform(0, 1 / (len_x * len_y), size=[len_y, len_x])
        b = np.random.normal(0, 0.1, size=len_y)
        sim = LinearSimulation(w, b, normal_scale=0.0, bin_scale=1.0, bin_rate=0, oneside=False)

        y = sim.predict(x)

        return x[:50000], y[:50000], x[50000:], y[50000:]


if __name__ == '__main__':
    from nn.model_deprecated import SequentialModel_v2
    from nn.layers_deprecated import FCLayer_v2
    from nn.activations_deprecated import Linear
    from nn.losses_deprecated import MseLoss
    from nn.optimizer_deprecated import GradientDecentOptimizer_v2

    model = SequentialModel_v2()
    model.add(FCLayer_v2(1, act=Linear()))
    model.compile(optimizer=GradientDecentOptimizer_v2(learn_rate=0.00002), loss=MseLoss(), metrics=[MseLoss()])

    len_x, len_y = 1024, 1
    x = np.random.uniform(0, 10, size=[60000, len_x])
    w = np.random.uniform(0, 1, size=[len_y, len_x])
    b = np.random.normal(0, 0.1, size=len_y)
    sim = LinearSimulation(w, b, normal_scale=0.0, bin_scale=1.0, bin_rate=0, oneside=False)

    y = sim.predict(x)

    tx, ty, ex, ey = x[:50000], y[:50000], x[50000:], y[50000:]

    model.fit(x=tx, y=ty, batch_size=10000, epochs=100)
    print(model.evaluate(ex, ey))
    print(np.sum(np.abs(w - model.NN[0].W)))