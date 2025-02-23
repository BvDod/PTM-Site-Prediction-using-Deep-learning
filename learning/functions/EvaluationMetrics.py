import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import torch
import comet_ml


from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

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


def get_evaluation_metrics(AA, y_true, y_output, y_pred, figures=True, species=False):
    """ Calculates all evaluation metrics and returns them as a dict"""

    if (not len(y_output.shape) > 1) or (y_output.shape[1] == 1):
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp+tn)/(fn+fp+tp+tn)
            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            precision = tp/(tp+fp)
            auc_roc = roc_auc_score(y_true, y_output)
            auc_pr = auc_pr_score(y_true, y_output)
            f1 = f1_score(y_true, y_pred)
            MCC = matthews_corrcoef(y_true, y_pred)
        except:
            accuracy, sensitivity, specificity, precision, auc_roc, auc_pr, f1, MCC = [0]*8
    
    elif species:

        y_true_PTM = y_true[:,[0,]]
        y_true_species = y_true[:,1:]
        y_output_PTM = y_output[:,[0,]]
        y_output_species = y_output[:,1:]
        y_pred_PTM = y_pred[:,[0,]]
        y_pred_species = y_pred[:,1:]
        

        tn, fp, fn, tp = confusion_matrix(y_true_PTM, y_pred_PTM).ravel()
        accuracy = (tp+tn)/(fn+fp+tp+tn)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        precision = tp/(tp+fp)
        auc_roc = roc_auc_score(y_true_PTM, y_output_PTM)
        auc_pr = auc_pr_score(y_true_PTM, y_output_PTM)
        f1 = f1_score(y_true_PTM, y_pred_PTM)
        MCC = matthews_corrcoef(y_true_PTM, y_pred_PTM)

        f1_species = f1_score(y_true_species, y_pred_species, average="macro")
        f1_scores = f1_score(y_true_species, y_pred_species, average=None)
        MCC_species = matthews_corrcoef(y_true_species, y_pred_species)

    else:
        f1 = f1_score(y_true, y_pred, average="macro")
        f1_scores = f1_score(y_true, y_pred, average=None)
        MCC = matthews_corrcoef(y_true, y_pred)


    if figures:
        roc_plot = plot_roc(y_true, y_output)
        pr_plot = plot_pr(y_true, y_output)


    if (not len(y_output.shape) > 1) or (y_output.shape[1] == 1):
        eval_metrics = {
            f"{AA} Validation Accuracy": accuracy,
            f"{AA} Validation Sensitivity": sensitivity,
            f"{AA} Validation Specificity": specificity,
            f"{AA} Validation Precision": precision,
            f"{AA} AUC ROC": auc_roc,
            f"{AA} AUC PR": auc_pr,
            f"{AA} MCC": MCC,
            f"{AA} F1": f1
        }
    
    elif species:
        eval_metrics = {
            f"{AA} Validation Accuracy": accuracy,
            f"{AA} Validation Sensitivity": sensitivity,
            f"{AA} Validation Specificity": specificity,
            f"{AA} Validation Precision": precision,
            f"{AA} AUC ROC": auc_roc,
            f"{AA} AUC PR": auc_pr,
            f"{AA} MCC": MCC,
            f"{AA} F1": f1,
            f"{AA} MCC (species)": MCC_species,
            f"{AA} F1": f1_species,
            }
        
        for i, e in enumerate(list(np.bincount(y_true_species.squeeze().astype(int)))):
            eval_metrics[f"{i}_sample count"] = e
        
        species_metrics = {}
        species_i = 0
        for i, sample_count in enumerate(list(np.bincount(y_true_species.squeeze().astype(int)))):
            if sample_count > 0:
                species_metrics[f"{AA} F1 (species={i})"] = f1_scores[species_i]
                species_i += 1
            else:
                species_metrics[f"{AA} F1 (species={i})"] = 0
        eval_metrics.update(species_metrics)





    else:
        eval_metrics = {
            f"{AA} MCC": MCC,
            f"{AA} F1": f1}
               
        for i, e in enumerate(list(np.bincount(y_true))):
            eval_metrics[f"{i}_sample count"] = e
        
        species_metrics = {}
        species_i = 0
        for i, sample_count in enumerate(list(np.bincount(y_true))):
            if sample_count > 0:
                species_metrics[f"{AA} F1 (species={i})"] = f1_scores[species_i]
                species_i += 1
            else:
                species_metrics[f"{AA} F1 (species={i})"] = 0
        eval_metrics.update(species_metrics)
        



    if figures:
        eval_figures = {
            f"{AA} ROC": roc_plot,
            f"{AA} PR": pr_plot
        }
    else:
        eval_figures = {}

    return eval_metrics, eval_figures

def logHpStudy(study, experiment, bestEvalMetricsAvg, bestEvalMetricsSTD):
    log_dict = {}
    best_trial = study.best_trial
    log_dict["EvalMetric (loss)"] = best_trial.value
    
    
    experiment.log_metrics(log_dict)
    experiment.log_metrics(best_trial.params, epoch=1)
    experiment.log_metrics(bestEvalMetricsAvg, epoch=1, prefix="avg")
    experiment.log_metrics(bestEvalMetricsSTD, epoch=1, prefix="std")

    plot_functions = [plot_edf, plot_intermediate_values, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice]
    figures = []
    for function in plot_functions:
        figure = function(study)
        experiment.log_figure(figure_name=f"{function.__name__}", figure=figure)
        # plt.close(figure)




def log_evaluation_metrics(experiment, epoch, metrics, figures):
    """Log metrics using tensorboard"""

    experiment.log_metrics(metrics, epoch=epoch)
    for name, figure in figures.items():
        experiment.log_figure(figure_name=name, figure=figure, step=epoch)
        plt.close(figure)

def log_evaluation_metrics_kFold(experiment, fold, metrics, figures):
    """Log metrics using tensorboard"""
    experiment.log_metrics(metrics, prefix=str(fold))
    for name, figure in figures.items():
        experiment.log_figure(figure_name=f"{name}_{fold}", figure=figure)
        plt.close(figure)

def log_kFold_average(experiment, average_dict, std_dict):

    experiment.log_metrics(average_dict, prefix="avg")
    experiment.log_metrics(std_dict, prefix="std")
    print(average_dict)
    print(std_dict)    
    return average_dict, std_dict


def CreateLoggers(parameters, net):
    """ Create a tensorboard logger instance which logs to the correct directory based on the parameters used"""

    aminoAcid = parameters["aminoAcid"]
    data_sample_mode, weight_decay = parameters["data_sample_mode"], parameters["weight_decay"]

    experiment = comet_ml.Experiment("1qqDK4gIHRXerCMtLeWAdGEdk", project_name="PTM-prediction")
    experiment.log_parameters(parameters)

    return experiment


def CreatekFoldLogger(parameters):
    """ Create a tensorboard logger instance which logs to the correct directory based on the parameters used"""
    name = parameters["Experiment Name"]
    experiment = comet_ml.Experiment("1qqDK4gIHRXerCMtLeWAdGEdk", project_name=f"Evaluation: {name}")
    experiment.log_parameters(parameters)
    print("results!")
    print(parameters)
    return experiment

def createHpTuningLogger(tuning_settings, parameters):
    name = parameters["Experiment Name"]
    experiment = comet_ml.Experiment("1qqDK4gIHRXerCMtLeWAdGEdk", project_name=f"HP-tuning: {name}")
    experiment.log_parameters(tuning_settings, prefix="tuning_")
    experiment.log_parameters(parameters, prefix="parameters_")
    return experiment





