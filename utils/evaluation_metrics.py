import scipy
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from collections import defaultdict
import numpy as np

from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score


class tabnet_AUC(Metric):
    """AUC to be used by TabNet library for multi-class case"""

    def __init__(self):  ##__init__
        self._name = 'AUC'
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        return auc


class tabnet_AUC_binary(Metric):
    """AUC to be used by TabNet library for the binary case"""

    def __init__(self):  ##__init__
        self._name = 'AUC_binary'
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score[:, 1])
        return auc


def get_best_roc_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    g_means = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(g_means)
    return thresholds[idx], g_means[idx]


def get_best_prec_recall_threshold(y_true, y_prob):
    # find precision recall graph
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # find F-score at each of its points
    f_score = [0 if (prec + rec == 0) else (2 * prec * rec) / (prec + rec) for (prec, rec) in zip(precision, recall)]
    # locate the index of the largest f score
    idx = np.argmax(f_score)
    return thresholds[idx], f_score[idx]


def compute_basic_metrics(y_true, y_score, y_pred):
    """Computes different metrics based on prediction vs ground truth"""
    metrics = {"precision": [], "sensitivity": [], "F1": [], "specificity": [], "NPV": []}
    classes = np.unique(y_true)

    if classes.shape[0] == 2:
        cur_true = y_true.astype(int)
        cur_pred = y_pred.astype(int).reshape(-1)
        class_w = 1
        tn, fp, fn, tp = confusion_matrix(cur_true, cur_pred).ravel()
        prec = 0 if tp + fp == 0 else tp / (tp + fp)
        rec = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        sps = 0 if tn + fp == 0 else tn / (tn + fp)
        npv = 0 if tn + fn == 0 else tn / (tn + fn)

        metrics["precision"].append(class_w * prec)
        metrics["sensitivity"].append(class_w * rec)
        metrics["F1"].append(class_w * f1)
        metrics["specificity"].append(class_w * sps)
        metrics["NPV"].append(class_w * npv)
        # metrics["FPR"].append(class_w * (1 - sps))
        # metrics["FNR"].append(class_w * fnr)

    for metric, rates in metrics.items():
        if len(classes) > 2:
            metrics[metric] = np.mean(rates).squeeze()
        else:
            metrics[metric] = np.sum(rates).squeeze()  # there sis only one value in the list so it is ok to use sum

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    if len(classes) > 2:
        metrics["AUC"] = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
    else:
        metrics["AUC"] = roc_auc_score(y_true, y_score)
    return metrics


def compute_metrics_with_confidence_estimation(metrics, confidence=0.95, return_str=False):
    "find metrics with confidence interval"
    arr = defaultdict(list)
    for dct in metrics:
        for k, v in dct.items():
            arr[k].append(v)

    for metric in arr:
        arr[metric] = mean_confidence_interval(arr[metric], confidence=confidence, return_str=return_str)

    return dict(arr)


def metrics2string(metrics, with_conf):
    # some utiltity function to store results
    n_digits = 2  # Round results to 2 digits
    res = ''
    for item in metrics.items():
        if with_conf is True:
            res += f"{item[0]}: {round(item[1]['Mean'], n_digits)} ({round(item[1]['Left bound'], n_digits)},{round(item[1]['Right bound'], n_digits)})\n"
        else:
            res += str(item) + '\n'
    return res


def mean_confidence_interval(a, sd=None, n=None, confidence=0.95, return_str=False):
    """
    Util: Given a list of values, returns a string of form:
    'mean (confidence, interval)'
    """
    if n is None:
        n = len(a)
        mean = np.mean(a)
        se = scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
        left = mean - h
        right = mean + h
    else:
        mean = a
        left, right = scipy.stats.norm.interval(alpha=confidence, loc=mean, scale=sd / np.sqrt(n))

    left = max(0, left)
    right = min(1, right)

    if return_str:
        return f'{mean:.2f} ({left:.2f}-{right:.2f})'
    else:
        return {'Mean': mean, 'Left bound': mean - h, 'Right bound': mean + h}
