from re import L
import numpy as np
from pytest import param

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import pprint
import optuna
from statistics import mean, stdev

from models.FC import FCNet
from torch.utils.tensorboard import SummaryWriter

from functions.EvaluationMetrics import get_evaluation_metrics, log_evaluation_metrics, CreateLoggers, CreatekFoldLogger, log_evaluation_metrics_kFold, log_kFold_average
from functions.DatasetHandling import split_training_test_data, loadData, createDatasets, CreateDataloaders
from models.firstLayer import firstLayer
import random

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def testModel(parameters, trial=None, logToComet=True, returnEvalMetrics=False, hyperparameterSeed=False):
    if hyperparameterSeed:
        parameters["random_state"] = 1 # Use static random state for assigning folds
    else:
        parameters["random_state"] = random.randint(0,((2**32)-parameters["CV_Repeats"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pprint.pprint(parameters)

    results = []
    for i in range(parameters["CV_Repeats"]):
        parameters["current_CV_Repeat"] = i + 1
    # Loop over the different folds for CV
        for fold in range(parameters["folds"]):
            model = FCNet
            net = model(device, parameters=parameters)
            optimizer = parameters["optimizer"](net.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"], eps=1e-6)
            net.to(device)
            
            trainloader, testloader, val_ratio, train_ratio = loadDataLoaders(parameters, fold)
            parameters["currentFold"] = fold
            best_metrics, best_figures = trainModel(trainloader, testloader, net, optimizer, device, parameters, val_ratio, train_ratio, trial=trial, logToComet=logToComet)
            
            results.append([best_metrics, best_figures])
            if not parameters["crossValidation"]:
                break
            parameters["random_state"] += 1
    
    metrics = [metric for metric, _ in results]
    average_dict = {}
    std_dict = {}
    for name in metrics[0].keys():
        average_dict[name] = mean([metric_dict[name] for metric_dict in metrics])
        if parameters["crossValidation"]:
            std_dict[name] = stdev([metric_dict[name] for metric_dict in metrics])
        else:
            std_dict[name] = 0

    

    # Log CV stats
    if logToComet:
        kFoldExperiment = CreatekFoldLogger(parameters)
        for fold, (best_metrics, best_figures) in enumerate(results):
            log_evaluation_metrics_kFold(kFoldExperiment, fold, best_metrics, best_figures)
            for name, figure in best_figures.items():
                figure.clf()
        avg_dict, std_dict = log_kFold_average(kFoldExperiment, average_dict, std_dict)
        
        if returnEvalMetrics:
            return average_dict, std_dict
        return avg_dict[parameters["ValidationMetric"]]
    
    for metric_dict, figure_dict in results:
        for name, figure in figure_dict.items():
            figure.clf()
    
    if returnEvalMetrics:
        return average_dict, std_dict
    print(average_dict)
    return average_dict[parameters["ValidationMetric"]]


def loadDataLoaders(parameters, fold):
    # Load AA of specific redundanctPercentage from disk
    X_neg, y_neg, X_pos, y_pos, n, tensor_dtype = loadData(parameters)

    # Split dataset into trainig and test set
    X_train_neg, y_train_neg, X_val_neg, y_val_neg = split_training_test_data(X_neg, y_neg, parameters, fold, tensor_dtype=tensor_dtype)
    X_train_pos, y_train_pos, X_val_pos, y_val_pos = split_training_test_data(X_pos, y_pos, parameters, fold, tensor_dtype=tensor_dtype)

    # Create torch Datasets, drop or dont drop redundant negative samples
    trainset, testset, train_weight, val_weight, n_train_pos, n_train_neg, train_ratio, val_ratio = createDatasets(
                                                                X_train_neg, y_train_neg, X_val_neg, y_val_neg,
                                                                X_train_pos, y_train_pos, X_val_pos, y_val_pos,
                                                                parameters)
    trainloader, testloader = CreateDataloaders(trainset, testset, n_train_pos, n_train_neg, parameters, train_weight)

    return trainloader, testloader, val_ratio, train_ratio


def trainModel(trainloader, testloader, net, optimizer, device, parameters, val_ratio, train_ratio, trial=None, logToComet=True):
    # Enable tensorbard logging
    if logToComet:
        experiment = CreateLoggers(parameters, net)
    else:
        experiment = None

    ###### The training loop
    t0 = time.time()
    max_batch = 0

    best_eval_score, best_eval_epoch = 100000, 0
    best_eval_metrics, best_eval_figures = None, None
    for epoch in range(parameters["epochs"]): 

        # Evaluate validation performance
        metrics, figures = evalValidation(testloader, net, device, val_ratio, experiment, epoch, parameters)

        if trial: 
            if (parameters["currentFold"] == 0) and (parameters["current_CV_Repeat"] == 1):
                trial.report(metrics[parameters["ValidationMetric"]], epoch)
                if (epoch >= 2) & trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if metrics[parameters["ValidationMetric"]] < best_eval_score:
                best_eval_score = metrics[parameters["ValidationMetric"]]
                best_eval_epoch = epoch
                if not best_eval_metrics is None:
                    for name, figure in best_eval_figures.items():
                        figure.clf()
                best_eval_metrics = metrics
                best_eval_figures = figures
        else:
            for name, figure in figures.items():
                figure.clf()



        if parameters["earlyStopping"]:
            if epoch - best_eval_epoch >= parameters["earlyStoppingPatience"]:
                print(f"Early stopping applied (best metric={best_eval_score})")
                break
        
        if metrics["Validation Loss"] > 10000:
            print(f"Exploding loss, terminate run (best metric={best_eval_score})")
            break

        if logToComet:
            context = experiment.train
        else:
            context = dummy_context_mgr

        # Actual training happens here
        with context():
            net.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)
                
                # Create criterion
                useSampleWeighting = (parameters["data_sample_mode"] == "weighted")
                if useSampleWeighting:
                    batch_weights = torch.ones(labels.shape, device=device, requires_grad=False)
                    batch_weights[labels == 0] = 1./train_ratio
                    criterion = parameters["loss_function"](weight=batch_weights)
                else:
                    criterion = parameters["loss_function"]()

                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Log average loss of epoch at last mini-batch of that epoch
                running_loss += loss.item()
                if i == max_batch:  
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/(max_batch+1)))
                    running_loss = 0.0
            max_batch = i

    print('Finished Training')
    t1 = time.time()
    # save_model(net)
    
    print(f"Total time taken: {t1 - t0}")
    return best_eval_metrics, best_eval_figures


def evalValidation(testloader, net, device, val_ratio, experiment, epoch, parameters):
    """Perform evaluation on validation set and log using tensorboard"""
    net.eval()

    if experiment:
        context = experiment.validate
    else:
        context = dummy_context_mgr

    with context():
        with torch.no_grad():
            y_pred, y_output, y_true = None, None, None
            weights_all_batches = None
            for data in testloader:
                features, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)

                y_output_batch = net(features)
                y_pred_batch = (y_output_batch > 0.5).int()

                useSampleWeighting = (parameters["data_sample_mode"] == "weighted")
                useSampleWeightingv = True

                if useSampleWeighting:
                    batch_weights = torch.ones(labels.shape, device=device)
                    batch_weights[labels == 0] = 1./val_ratio
                    
                if y_pred is None:
                    y_pred, y_output = y_pred_batch, y_output_batch
                    y_true = labels
                    if useSampleWeighting:
                        weights_all_batches = batch_weights
                else:
                    y_pred = torch.cat((y_pred, y_pred_batch), axis=0)
                    y_output = torch.cat((y_output, y_output_batch), axis=0)
                    y_true = torch.cat((y_true, labels), axis=0)
                    if useSampleWeighting:
                        weights_all_batches = torch.cat([weights_all_batches, batch_weights])
        
            if useSampleWeighting:     
                criterion = parameters["loss_function"](weight=weights_all_batches)
            else:
                criterion = parameters["loss_function"]()

            loss = criterion(y_output, y_true)

            eval_metrics, eval_figures = get_evaluation_metrics(y_true.detach().cpu().numpy(), y_output.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            eval_metrics["Validation Loss"] = loss.detach().item()
            if experiment:
                log_evaluation_metrics(experiment, epoch, eval_metrics, eval_figures)

            return eval_metrics, eval_figures


def save_model(net):
    if net.model_parameters["embeddingType"] == "protBert":
        net.layers.pop(0)
    torch.save(net.state_dict(), "model.txt")


def load_model(filestring):
    net = torch.save(filestring)
    if net.embeddingType == "protBert":
        net.firstLayer = firstLayer(net.device, net.embeddingType)
    return net
    

if __name__ == "__main__":

    parameters = { 
        # Training parameters
        "gpu_mode": True,
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.003,
        "aminoAcid": "Sumoylation",
        "test_data_ratio": 0.1,
        "data_sample_mode": "undersample",
        "loss_function": nn.BCEWithLogitsLoss,
        "optimizer": optim.Adam,
        "crossValidation": True,
        "folds": 5,
        "earlyStopping": True,
        "earlyStoppingMetric": "AUC ROC",
        "earlyStoppingPatience": 10,
        
        # Model parameters
        "weight_decay": 10,
        "embeddingType": "adaptiveEmbedding",
        "embeddingSize": 2,
        "CNN_layers": 2,
        "CNN_filters": 32,
        "CNN_dropout": 0.2,
        "LSTM_layers": 2,
        "LSTM_hidden_size": 32,
        "LSTM_dropout": 0.2,
        "FC_layers": 2,
        "FC_layer_size": 64,
        "FC_dropout": 0.65, 
        } 
                      

    testModel(parameters)
    AAs = ["Acetylation", "Hydroxylation-K", "Hydroxylation-P", "Methylation-K", "Methylation-R", "N-linked Glycosylation", "O-linked Glycosylation", "Phosphorylation-['S', 'T']", "Phosphorylation-Y", "Pyrrolidone carboxylic acid", "S-palmitoylation-C", "Sumoylation", "Ubiquitination"]
    for aa in AAs:
        parameters["aminoAcid"] = aa
        testModel(parameters)
