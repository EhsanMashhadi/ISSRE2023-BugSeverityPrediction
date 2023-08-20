import os.path
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve, confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

labels = {0: "Critical, Blocker", 1: "Major, High", 2: "Medium", 3: "Low, Trivial, Minor"}
colors = cycle(["#C0392B", "#E67E22", "#F1C40F", "#71B5A0"])
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def evaluatelog_result(y_true, y_prediction, prob, model_name, logger):
    logger.info("************ " + model_name + " ************")
    eval_result = evaluate_result(y_true=y_true, y_prediction=y_prediction, prob=prob)
    plot_roc(y_test=y_true, prob=prob, model_name=model_name)
    plot_precision_recall(y_test=y_true, prob=prob, model_name=model_name)
    plot_confusionmatrix(y_true, y_prediction, model_name=model_name)
    for key in sorted(eval_result.keys()):
        logger.info("  %s = %s", key, str(eval_result[key]))


def evaluate_result(y_true, y_prediction, prob):
    f1_weighted = f1_score(y_true, y_prediction, average='weighted')
    f1_per_class = f1_score(y_true, y_prediction, average=None)
    accuracy = accuracy_score(y_true, y_prediction)
    precision = precision_score(y_true, y_prediction, average='weighted')
    recall = recall_score(y_true, y_prediction, average='weighted')
    roc_uac = roc_auc_score(y_true, prob, average='weighted', multi_class='ovo')
    mcc = matthews_corrcoef(y_true, y_prediction)

    eval_result = {
        "eval_f1": float(f1_weighted),
        "eval_f1_perclass": f1_per_class,
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_ROC-UAC": float(roc_uac),
        "eval_mcc": float(mcc)
    }

    return eval_result


def plot_roc(y_test, prob, model_name):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], prob[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="Class {}, AUC = {:0.2f} ({})".format(i, roc_auc[i], labels[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR", fontsize=18)
    plt.ylabel("TPR", fontsize=18)
    plt.title("ROC Curve of {}".format(model_name), fontsize=18, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best")

    plt.savefig(os.path.join("log", "roc_{}".format(model_name)))
    plt.clf()


def plot_precision_recall(y_test, prob, model_name):
    # precision recall curve
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))
    precision = dict()
    recall = dict()
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], prob[:, i])
        plt.plot(recall[i], precision[i], lw=2, color=color, label='Class {} ({})'.format(i, labels[i]))

    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.legend(loc="best")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Precision-Recall Curve of {}".format(model_name), fontsize=18, fontweight="bold")
    plt.savefig(os.path.join("log", "precision_recall_{}".format(model_name)))
    plt.clf()


def plot_confusionmatrix(y_test, result, model_name):
    cm = confusion_matrix(y_test, result)
    cm_df = pd.DataFrame(cm,
                         index=labels,
                         columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap="Dark2_r", annot_kws={"fontweight": "bold", "fontsize": 18},
                linewidths=0.003)
    plt.ylabel('Actual Values', fontsize=22, fontweight="bold")
    plt.xlabel('Predicted Values', fontsize=22, fontweight="bold")
    plt.xticks(fontsize=18, fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")
    plt.suptitle("Confusion Matrix of {}".format(model_name), fontsize=18, fontweight="bold")
    plt.savefig(os.path.join("log", "confusion_matrix_{}".format(model_name)))
    plt.clf()
