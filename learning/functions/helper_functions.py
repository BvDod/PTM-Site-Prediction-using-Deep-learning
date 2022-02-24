import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch

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


def indexListToOneHot(input_array):
    """This function converts an array of categorical features to an array of one-hot represented features"""

    OneHotVariables = 27
    samples, columns = input_array.shape

    output_array = np.zeros((samples, OneHotVariables*columns))
    for i in range(samples):
        for j in range(columns):
            output_array[i, (j*OneHotVariables) + input_array[i,j]] = 1
    return output_array


def get_folder_name(AA, redundancy):
    data_dir = "dataset/data/learningData/balanced/"
    folder_name = f"{data_dir}oneHot_{str(redundancy)}/{AA}"
    return folder_name


def split_training_test_data(X, y, test_data_ratio=0.1, tensor_dtype=torch.float):
    """ Splits the training and test data into fractions NOTE: NO SHUFFLING, ALREADY DID THAT"""
    
    n = len(y)
    X_val = X[:int(n*test_data_ratio),:]
    X_val = torch.tensor(X_val, dtype=tensor_dtype)

    y_val = y[:int(n*test_data_ratio)]
    y_val = torch.tensor(y_val, dtype=torch.float)

    X_train = X[int(n*test_data_ratio):,:]
    X_train = torch.tensor(X_train, dtype=tensor_dtype)

    y_train = y[int(n*test_data_ratio):]
    y_train = torch.tensor(y_train, dtype=torch.float)

    return X_train, y_train, X_val, y_val


def get_evaluation_metrics( y_true, y_output, y_pred):
    """ Calculates all evaluation metrics and returns them as a dict"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp+tn)/(fn+fp+tp+tn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    auc = roc_auc_score(y_true, y_output)
    roc_plot = plot_roc(y_true, y_output)

    eval_metrics = {
        "Validation Accuracy": accuracy,
        "Validation Sensitivity": sensitivity,
        "Validation Specificity": specificity,
        "Validation Precision": precision,
        "AUC ROC": auc,
    }

    eval_figures = {
        "ROC": roc_plot
    }

    return eval_metrics, eval_figures


def log_evaluation_metrics(tb, epoch, metrics, figures):
    """Log metrics using tensorboard"""

    for name, metric in metrics.items():
        tb.add_scalar(name, metric, epoch)
    for name, figure in figures.items():
        tb.add_figure(name, figure, epoch)




