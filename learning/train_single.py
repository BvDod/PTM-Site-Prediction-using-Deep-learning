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
import gc

from models.FC import FCNet
from torch.utils.tensorboard import SummaryWriter

from functions.EvaluationMetrics import get_evaluation_metrics, log_evaluation_metrics, CreateLoggers, CreatekFoldLogger, log_evaluation_metrics_kFold, log_kFold_average
from functions.DatasetHandling import split_training_test_data, loadData, createDatasets, CreateDataloaders
from functions.MultiTaskLoader import MultiTaskLoader, MultiTaskLossWrapper
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


def testModel(parameters, trial=None, logToComet=True, returnEvalMetrics=False, hyperparameterSeed=False,device_id="all"):
    if hyperparameterSeed:
        parameters["random_state"] = 1 # Use static random state for assigning folds
    else:
        parameters["random_state"] = random.randint(0,((2**32)-parameters["CV_Repeats"]))
    if device_id == "all":
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    pprint.pprint(parameters)

    results = []
    for i in range(parameters["CV_Repeats"]):
        parameters["current_CV_Repeat"] = i + 1
    # Loop over the different folds for CV
        for fold in range(parameters["folds"]):
            model = FCNet

            
            net = model(device, parameters=parameters)
            if device_id == "all" and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model)
            parameter_dicts = [
                {'params': net.layers.parameters()}
            ]
            if (len(parameters["aminoAcid"]) > 1) and "TaskWeightDecay" in parameters:
                parameter_dicts = parameter_dicts + [{'params': net.heads[i].parameters(), "weight_decay": parameters["TaskWeightDecay"][i]} for i in range(len(parameters["TaskWeightDecay"]))]
            else:
                parameter_dicts.append({'params': net.heads.parameters()})

            optimizer = parameters["optimizer"](parameter_dicts, lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"], eps=1e-6)
            net.to(device)
            
            trainloader, testloader, val_ratios, train_ratios = loadDataLoaders(parameters, fold)


            parameters["currentFold"] = fold
            best_metrics, best_figures = trainModel(trainloader, testloader, net, optimizer, device, parameters, val_ratios, train_ratios, trial=trial, logToComet=logToComet)
            
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
    trainloaders, testloaders = [], []
    val_ratios, train_ratios = [], []

    loaded_data_neg = []
    loaded_data_pos = []

    biggest_dataset_amount = None
    smallest_dataset_amount = None
    
    sample_sizes = []
    for i, aminoAcid in enumerate(parameters["aminoAcid"]):
        X_neg, y_neg, X_pos, y_pos, n, tensor_dtype = loadData(parameters, aminoAcid)
        X_train_neg, y_train_neg, X_val_neg, y_val_neg = split_training_test_data(X_neg, y_neg, parameters, fold, tensor_dtype=tensor_dtype)
        X_train_pos, y_train_pos, X_val_pos, y_val_pos = split_training_test_data(X_pos, y_pos, parameters, fold, tensor_dtype=tensor_dtype)
        
        del X_neg
        del y_neg
        del X_pos
        del y_pos
        
        loaded_data_neg.append([X_train_neg, y_train_neg, X_val_neg, y_val_neg])
        loaded_data_pos.append([X_train_pos, y_train_pos, X_val_pos, y_val_pos])

        if (parameters["data_sample_mode"][i] == "balanced") or (parameters["data_sample_mode"][i] == "undersample"):
            samples = len(y_train_pos) * 2
        if parameters["data_sample_mode"][i] == "oversample":
            samples = len(y_train_neg) * 2
        if parameters["data_sample_mode"][i] == "weighted":
            samples = len(y_train_pos) + len(y_train_pos)
        
        sample_sizes.append(samples)
        if biggest_dataset_amount is None or biggest_dataset_amount < samples:
            biggest_dataset_amount = samples
        if smallest_dataset_amount is None or smallest_dataset_amount > samples:
            smallest_dataset_amount = samples

    if "MultiTask_sample_method" in parameters:
        if parameters["MultiTask_sample_method"] == "undersample":
            dataloader_samples = smallest_dataset_amount
        
        elif parameters["MultiTask_sample_method"] == "oversample":
            dataloader_samples = biggest_dataset_amount
        
        elif type(parameters["MultiTask_sample_method"]) == int:
            dataloader_samples = parameters["MultiTask_sample_method"]
        
        elif parameters["MultiTask_sample_method"] == "balanced":
            pass
    else:
        dataloader_samples = biggest_dataset_amount

    sample_weights = []
    for i, aminoAcid in enumerate(parameters["aminoAcid"]):
        
        # Split dataset into trainig and test set
        X_train_neg, y_train_neg, X_val_neg, y_val_neg = loaded_data_neg[i]
        X_train_pos, y_train_pos, X_val_pos, y_val_pos = loaded_data_pos[i]

        # Create torch Datasets, drop or dont drop redundant negative samples
        trainset, testset, train_weight, val_weight, n_train_pos, n_train_neg, train_ratio, val_ratio = createDatasets(
                                                                    X_train_neg, y_train_neg, X_val_neg, y_val_neg,
                                                                    X_train_pos, y_train_pos, X_val_pos, y_val_pos,
                                                                    reduceNegativeSamples=(parameters["data_sample_mode"][i] == "balanced"))
        del X_train_neg
        del y_train_neg
        del X_val_neg
        del y_val_neg
        del X_train_pos
        del y_train_pos
        del X_val_pos
        del y_val_pos

        if ("MultiTask_sample_method" in parameters) and parameters["MultiTask_sample_method"] == "balanced":
            samples_per_batch = math.ceil((sample_sizes[i] / sum(sample_sizes)) * parameters["batch_size"])
            if samples_per_batch < 1:
                samples_per_batch = 1
            dataloader_samples = int((sum(sample_sizes) / parameters["batch_size"]) * samples_per_batch)
            batch_size = samples_per_batch
            print(batch_size, dataloader_samples)
            
        else:
            batch_size = parameters["batch_size"]//len(parameters["aminoAcid"])
        
        sample_weights.append(sample_sizes[i]/max(sample_sizes))    
        trainloader, testloader = CreateDataloaders(trainset, testset, n_train_pos, n_train_neg, parameters, train_weight, parameters["data_sample_mode"][i], dataloader_samples, batch_size)
        
        trainloaders.append(trainloader)
        testloaders.append(testloader)

        val_ratios.append(val_ratio)
        train_ratios.append(train_ratio)
    
    trainloader = MultiTaskLoader(trainloaders)
    parameters["sample_weights"] = sample_weights
    return trainloader, testloaders, val_ratios, train_ratios


def trainModel(trainloader, testloaders, net, optimizer, device, parameters, val_ratio, train_ratio, trial=None, logToComet=True):
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

    if parameters["UseUncertaintyBasedLoss"]:
        uncertaintyWrapper = MultiTaskLossWrapper(len(parameters["aminoAcid"]))

    for epoch in range(parameters["epochs"]): 
        gc.collect()

        # Evaluate validation performance
        if not epoch == 0:
            metrics, figures = evalValidation(testloaders, net, device, val_ratio, experiment, epoch, parameters)

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
            
            if metrics[parameters["ValidationMetric"]] > 10000:
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
                inputs, labels, task_indexes = data[0].to(device), data[1].reshape((-1,1)).to(device), data[2]
                
                useSampleWeighting = (parameters["data_sample_mode"] == "weighted")

                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = net(inputs, task_indexes)

                losses = []
                for task in range(len(task_indexes[:-1])):
                    outputs_task = outputs[task_indexes[task]: task_indexes[task+1]]
                    labels_task = labels[task_indexes[task]: task_indexes[task+1]]
                    if useSampleWeighting:     
                        task_weights = torch.ones(outputs_task.shape, device=device)
                        task_weights[labels_task == 0] = 1./(train_ratio[task]/2)
                        task_weights[labels_task == 1] = (train_ratio[task]/2)
                        criterion = parameters["loss_function"](weight=task_weights)
                    else:
                        criterion = parameters["loss_function"]()

                    loss = criterion(outputs_task, labels_task)
                    if parameters["useLrWeight"]:
                        if parameters["MultiTask_sample_method"] == "balanced":
                            weight = (task_indexes[task+1] - task_indexes[task]) / parameters["batch_size"]
                        else:
                            weight = parameters["sample_weights"][task]
                    else:
                        weight = 1
                    losses.append(weight * loss)

                if parameters["UseUncertaintyBasedLoss"]:
                    loss = uncertaintyWrapper(losses)
                else:
                    loss = sum(losses)
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


def evalValidation(testloaders, net, device, val_ratio, experiment, epoch, parameters):
    """Perform evaluation on validation set and log using tensorboard"""
    net.eval()

    if experiment:
        context = experiment.validate
    else:
        context = dummy_context_mgr

    with context():
        with torch.no_grad():
            y_pred, y_output, y_true = [], [], []

            for i, testloader in enumerate(testloaders):
                y_pred_task, y_output_task, y_true_task = [], [], []
                for data in testloader:
                    features, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)
                    task_indexes = [0, len(labels)]
                    
                    y_output_batch = net(features, task_indexes)
                    y_pred_batch = (y_output_batch > 0.5).int()

                    useSampleWeighting = (parameters["data_sample_mode"] == "weighted")
                    useSampleWeighting = True
    
                    y_pred_task.append(y_pred_batch)
                    y_output_task.append(y_output_batch)
                    y_true_task.append(labels)
                y_pred.append(torch.cat(y_pred_task))
                y_output.append(torch.cat(y_output_task))
                y_true.append(torch.cat(y_true_task))
            
                        

            losses = []
            eval_metrics_total = {}
            eval_figures_total = {}
            for task in range(len(testloaders)):
                AA = parameters['aminoAcid'][task]
                if useSampleWeighting:     
                    task_weights = torch.ones(y_true[task].shape, device=device)
                    task_weights[y_true[task] == 0] = 1./(val_ratio[task]/2)
                    task_weights[y_true[task] == 1] = (val_ratio[task]/2)

                    criterion = parameters["loss_function"](weight=task_weights)
                else:
                    criterion = parameters["loss_function"]()

                loss = criterion(y_output[task], y_true[task])
                losses.append(loss)

                eval_metrics, eval_figures = get_evaluation_metrics(AA, y_true[task].detach().cpu().numpy(), y_output[task].detach().cpu().numpy(), y_pred[task].detach().cpu().numpy())
                eval_metrics[f"Validation Loss ({AA})"] = loss.detach().item()
                eval_metrics_total.update(eval_metrics)
                eval_figures_total.update(eval_figures)

            total_loss = sum(losses)
            eval_metrics_total["Validation Loss (total)"] = total_loss.detach().item()
            
            if experiment:
                log_evaluation_metrics(experiment, epoch, eval_metrics_total, eval_figures_total)

            return eval_metrics_total, eval_figures_total

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
        "epochs": 200,
        "batch_size": 2048,
        "learning_rate": 0.0005,
        "test_data_ratio": 0.2,
        "data_sample_mode": "oversample",
        "crossValidation": True,
        "loss_function": nn.BCELoss,
        "optimizer": optim.AdamW,
        "folds": 1,
        "earlyStopping": True,
        "ValidationMetric": "Validation Loss (total)",
        "earlyStoppingPatience": 50,
        "CV_Repeats": 1,
        "Experiment Name": "Model architecture - added max, ranges, bceloss",


        # Model parameters
        "weight_decay": 2.5,
        "embeddingType": "adaptiveEmbedding",
        "LSTM_layers": 1,
        "LSTM_hidden_size": 32,
        "LSTM_dropout": 0,
        "MultiTask": False,

        "MultiTask_sample_method": "balanced",
        "UseUncertaintyBasedLoss": False,
        "useLrWeight": False,

        "CNNType": "Adapt",
        "FCType": "Adapt",

        "layerToSplitOn": "FC"
        }
                      

    parameters["aminoAcid"] = ["Hydroxylation-P"]
    parameters["data_sample_mode"] = ["balanced"] * 13
    # parameters["TaskWeightDecay"] = [0.1] * 2
    testModel(parameters)
