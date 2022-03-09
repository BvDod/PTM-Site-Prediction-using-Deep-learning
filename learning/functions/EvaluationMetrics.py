import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import matthews_corrcoef
import torch

from torch.utils.tensorboard import SummaryWriter


matplotlib.use('Agg')

def plot_roc(y_true, y_pred):
    """ This function returns a mpl figure of the ROC-curve"""

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig = plt.figure(dpi=128)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fig


def plot_pr(y_true,y_pred):
    """ This function returns a mpl figure of the PR-curve"""

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fig = plt.figure(dpi=128)
    plt.title('Precision-Recall Curve')
    plt.plot(recall, precision, 'b')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    return fig

   
def auc_pr_score(y_true, y_pred):
    """ Calculates AUC-PR score"""

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return auc_pr


def get_evaluation_metrics(y_true, y_output, y_pred):
    """ Calculates all evaluation metrics and returns them as a dict"""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp+tn)/(fn+fp+tp+tn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    auc_roc = roc_auc_score(y_true, y_output)
    roc_plot = plot_roc(y_true, y_output)
    pr_plot = plot_pr(y_true, y_output)
    auc_pr = auc_pr_score(y_true, y_output)
    MCC = matthews_corrcoef(y_true, y_pred)

    eval_metrics = {
        "Validation Accuracy": accuracy,
        "Validation Sensitivity": sensitivity,
        "Validation Specificity": specificity,
        "Validation Precision": precision,
        "AUC ROC": auc_roc,
        "AUC PR": auc_pr,
        "MCC": MCC
    }

    eval_figures = {
        "ROC": roc_plot,
        "PR": pr_plot
    }

    return eval_metrics, eval_figures


def log_evaluation_metrics(tb, epoch, metrics, figures):
    """Log metrics using tensorboard"""

    for name, metric in metrics.items():
        tb.add_scalar(name, metric, epoch)
    for name, figure in figures.items():
        tb.add_figure(name, figure, epoch)


def CreateTensorBoardLogger(parameters, net):
    """ Create a tensorboard logger instance which logs to the correct directory based on the parameters used"""

    aminoAcid, redundancyPercentage, = parameters["aminoAcid"], parameters["redundancyPercentage"]
    data_sample_mode, weight_decay = parameters["data_sample_mode"], parameters["weight_decay"]

    if parameters["embeddingType"] == "embeddingLayer":
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.firstLayer.embeddingSize}/{net.FC_layer_sizes}/{data_sample_mode}/{weight_decay}/"
    else:
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.FC_layer_sizes}/{data_sample_mode}/{weight_decay}/"
    
    i = 1
    while os.path.exists(f"{log_dir}{i}/"):
        i += 1
    log_dir = f"{log_dir}{i}/"
    
    tb = SummaryWriter(log_dir=log_dir)
    return tb




