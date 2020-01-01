from abc import ABCMeta, abstractmethod, ABC

import numpy as np


class IMetrics(metaclass=ABCMeta):

    @abstractmethod
    def metric(self, y, label):
        """
            Calculate metrics value
        :return: Scala: single metrics value
        """
        pass

    @abstractmethod
    def description(self):
        """
            Official name for this metric.
        :return:
        """
        pass


def eval_class(prediction, threshold):

    return np.asarray(prediction > threshold, dtype=int)


def get_tp(prediction, label):

    return np.sum(label & prediction)


def get_fp(prediction, label):

    return np.sum((label == 0).astype('int') & prediction)


def get_tn(prediction, label):

    return np.sum((label == 0).astype('int') & (prediction == 0).astype('int'))


def get_fn(prediction, label):

    return np.sum(label & (prediction == 0).astype('int'))


def get_tpr(prediction, label):

    return get_tp(prediction, label) / (get_tp(prediction, label) + get_fn(prediction, label))


def get_fpr(prediction, label):

    return get_fp(prediction, label) / (get_fp(prediction, label) + get_tn(prediction, label))


def get_tnr(prediction, label):

    return get_tn(prediction, label) / (get_fp(prediction, label) + get_tn(prediction, label))


def get_far(prediction, label):

    return get_fp(prediction, label) / (get_fp(prediction, label) + get_tn(prediction, label))


def get_frr(prediction, label):

    return get_fn(prediction, label) / (get_tp(prediction, label) + get_fn(prediction, label))


def get_far_with_threshold(prediction, label, step=0.1):

    r = [get_far(eval_class(prediction, threshold), label) for threshold in np.arange(0, 1+step, step)]
    return np.asarray(r)


def get_frr_with_threshold(prediction, label, step=0.1):

    r = [get_frr(eval_class(prediction, threshold), label) for threshold in np.arange(0, 1+step, step)]
    return np.asarray(r)


def get_roc(prediction, label, step=0.1):

    x = np.asarray([get_fpr(eval_class(prediction, threshold), label) for threshold in np.arange(0, 1+step, step)])
    y = np.asarray([get_tpr(eval_class(prediction, threshold), label) for threshold in np.arange(0, 1+step, step)])

    return x, y


def linear_solve(x, y):

    r = []

    for i in range(len(x) - 1):
        if y[i] * y[i+1] < 0:
            if y[i] > 0:
                high = y[i]
                low = -1 * y[i+1]
                start = x[i]
                end = x[i+1]
            else:
                high = y[i+1]
                low = -1 * y[i]
                start = x[i+1]
                end = x[i]

            r.append((low * start + high * end) / (high + low))
        if y[i] == 0:
            r.append(x[i])

    if y[-1] == 0:
        r.append(x[-1])

    return r


def get_definite_integral(x, y):

    sum_ = 0
    for i in range(len(x) - 1):
        sum_ += y[i] * (x[i + 1] - x[i])

    return sum_


def get_eer(prediction, label, accuracy=0.001):

    far = get_far_with_threshold(prediction, label, accuracy)
    frr = get_frr_with_threshold(prediction, label, accuracy)

    er = far - frr
    threshold = linear_solve(np.arange(0, 1+accuracy, accuracy), er)[0]

    eer_ = get_far(eval_class(prediction, threshold), label)

    return eer_


def get_auc(prediction, label, accuracy=0.001):

    x, y = get_roc(prediction, label, accuracy)
    sum_ = get_definite_integral(x, y)
    sum_ = sum_ if sum_ < 1.0 else 1.0
    return sum_


class BinaryAccuracy(IMetrics):
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
        result = np.sum(y == label)
        return result

    def description(self):
        return 'accuracy'


class CategoricalAccuracy(IMetrics):
    """
        Categorical Accuracy metric.
        Use one-hot vector as label.

        Metric can be used in MLR, CNN.
    """

    def __init__(self, threshold=0.5, threshold_vector=None):
        if threshold_vector is not None:
            self.Threshold = threshold_vector
        else:
            self.Threshold = threshold

    def metric(self, y, label):
        y = np.floor(y + self.Threshold).astype('int')
        label = label.astype('int')
        result = np.sum(y & label)
        return result

    def description(self):
        return 'accuracy'


class RelativeError(IMetrics):
    """
        Relative error metric.
        Use numerical result as prediction.

        Metric can be used in OLR, FCN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        result = np.abs(y - label)
        return result.sum()

    def description(self):
        return 'relative error'


class MeanSqureError(IMetrics):
    """
        Mean squre error metric.
        Use numerical result as prediction.

        Metric can be used in OLR, FCN.
    """

    def __init__(self):
        pass

    def metric(self, y, label):
        result = np.square(y - label)
        return result.sum()

    def description(self):
        return 'MSE'


class EqualErrorRate(IMetrics):
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


class TruePositive(IMetrics):
    """
        True positive.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_tp(eval_class(y, self.Threshold), label)

    def description(self):
        return 'TP'


class FalsePositive(IMetrics):
    """
        False positive.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_fp(eval_class(y, self.Threshold), label)

    def description(self):
        return 'FP'


class TrueNegative(IMetrics):
    """
        True negative.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_tn(eval_class(y, self.Threshold), label)

    def description(self):
        return 'TN'


class FalseNegative(IMetrics):
    """
        False negative.
    """

    def __init__(self, threshold=0.5):
        self.Threshold = threshold

    def metric(self, y, label):
        return get_fn(eval_class(y, self.Threshold), label)

    def description(self):
        return 'FN'


class AreaUnderCurve(IMetrics):
    """
        AUC
    """

    def __init__(self, accuracy=0.01):
        self.Accuracy = accuracy

    def metric(self, y, label):
        return get_auc(y, label, self.Accuracy)

    def description(self):
        return 'AUC'

