import numpy as np


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
