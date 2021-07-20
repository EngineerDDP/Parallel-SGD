from nn.metric.metrichelper import *
from nn.metric.interface import IMetric


class BinaryAccuracy(IMetric):
    """
        Binary accuracy metric.
        Default classification threshold is 0.5.
        for each sample, its class either 1 or 0.

        Metric can be used in SVM, LR.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        y = np.floor(y + self.Threshold)
        result = np.mean(y == label)
        return result

    def description(self):
        return 'accuracy'


class CategoricalAccuracy(IMetric):
    """
        Categorical Accuracy metric.
        Use one-hot vector as label.

        Metric can be used in MLR, CNN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        y = (y == y.max(axis=1).reshape([-1, 1])).astype('int')
        label = label.astype('int')
        result = np.sum(y & label) / len(y)
        return result

    def description(self):
        return 'accuracy'


class RelativeError(IMetric):
    """
        Relative error metric.
        Use numerical result as prediction.

        Metric can be used in OLR, FCN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        result = np.abs(y - label)
        return result.mean()

    def description(self):
        return 'relative error'


class MeanSquareError(IMetric):
    """
        Mean squre error metric.
        Use numerical result as prediction.

        Metric can be used in OLR, FCN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        result = np.sum(np.square(y - label), axis=1)
        return result.mean()

    def description(self):
        return 'MSE'


class RelativeMeanSquareError(IMetric):
    """
        RMSE
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        res = np.sum(np.square(y - label), axis=1)
        return np.mean(np.sqrt(res))

    def description(self):
        return 'RMSE'


class EqualErrorRate(IMetric):
    """
        Equal error rate metric.
        Use interpolation method.

        Metric can only be used in 1/0 classification.
    """

    def __init__(self, inter_accuracy=0.001):
        self.Accuracy = inter_accuracy

    def metric(self, y, label):
        return get_eer(y, label, self.Accuracy)

    def description(self):
        return 'EER'


class TruePositive(IMetric):
    """
        True positive.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_tp(eval_class(y, self.Threshold), label)

    def description(self):
        return 'TP'


class FalsePositive(IMetric):
    """
        False positive.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_fp(eval_class(y, self.Threshold), label)

    def description(self):
        return 'FP'


class TrueNegative(IMetric):
    """
        True negative.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_tn(eval_class(y, self.Threshold), label)

    def description(self):
        return 'TN'


class FalseNegative(IMetric):
    """
        False negative.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_fn(eval_class(y, self.Threshold), label)

    def description(self):
        return 'FN'


class AreaUnderCurve(IMetric):
    """
        AUC
    """

    def __init__(self, accuracy=0.01):
        self.Accuracy = accuracy

    def metric(self, y, label):
        return get_auc(y, label, self.Accuracy)

    def description(self):
        return 'AUC'
