import numpy as np
from sklearn import metrics


def roc_curve(y_true, y_score, pos_label=1):
    return metrics.roc_curve(y_true, y_score, pos_label=pos_label)


def ROC_AUC(fpr, tpr):
    return metrics.auc(fpr, tpr)


def EER(fpr, tpr):
    fnr = 1 - tpr

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def FNR_1_FPR(fpr, tpr):
    # the fnr then fpr == 1%
    fnr = 1 - tpr
    fnr_1_fpr = fpr[np.nanargmin(np.absolute((fnr - 0.01)))]
    return fnr_1_fpr


def FPR_1_FNR(fpr, tpr):
    # the fpr then fnr == 1%
    fnr = 1 - tpr
    fpr_1_fnr = fnr[np.nanargmin(np.absolute((fpr - 0.01)))]
    return fpr_1_fnr


def compute_thresholds(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    # the threshold of fpr == 1%
    fpr_1_threshold = threshold[np.nanargmin(np.absolute((fpr - 0.01)))]
    # the fnr then fpr == 1%
    fpr_1_fnr = fnr[np.nanargmin(np.absolute((fpr - 0.01)))]

    # the threshold of fnr == 1%
    fnr_1_threshold = threshold[np.nanargmin(np.absolute((fnr - 0.01)))]
    # the fpr then fnr == 1%
    fnr_1_fpr = fpr[np.nanargmin(np.absolute((fnr - 0.01)))]

    return eer_threshold, eer, fpr_1_threshold, fpr_1_fnr, fnr_1_threshold, fnr_1_fpr


def get_f1(pred_labels, labels):
    TP = (pred_labels[labels == 1] == 1).sum()
    FP = (pred_labels[labels == 0] == 1).sum()
    TN = (pred_labels[labels == 0] == 0).sum()
    FN = (pred_labels[labels == 1] == 0).sum()

    if FP == TN == 0:
        fpr = 1
    else:
        fpr = FP / (FP + TN)

    fnr = FN / (FN + TP)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1, fpr, fnr, precision, recall


# Only registered metric can be used.
REGISTERED_METRICS = {
    "ROC_AUC": ROC_AUC,
    "EER": EER,
    "FNR_1_FPR": FNR_1_FPR,
    "FPR_1_FNR": FPR_1_FNR
}
