import numpy as np

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import pprint


from models.FC import FCNet
from torch.utils.tensorboard import SummaryWriter

from functions.EvaluationMetrics import get_evaluation_metrics, log_evaluation_metrics, CreateLoggers, CreatekFoldLogger, log_evaluation_metrics_kFold, log_kFold_average
from functions.DatasetHandling import split_training_test_data, loadData, createDatasets, CreateDataloaders
from models.firstLayer import firstLayer
import random

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler






def testModel(parameters):
    parameters["random_state"] = random.randint(0,(2**32)-1)
    pprint.pprint(parameters)

    model = parameters["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = []
    for fold in range(parameters["folds"]):
        model = parameters["model"]
        net = model(device, peptide_size=33, embeddingType=parameters["embeddingType"])
        optimizer = parameters["optimizer"](net.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
        net.to(device)
        
        trainloader, testloader, val_ratio, train_ratio = loadDataLoaders(parameters, fold)
        best_metrics, best_figures = trainModel(trainloader, testloader, net, optimizer, device, parameters, val_ratio, train_ratio)
        results.append([best_metrics, best_figures])
        if not parameters["crossValidation"]:
            break
    
    kFoldExperiment = CreatekFoldLogger(parameters)
    for fold, (best_metrics, best_figures) in enumerate(results):
        log_evaluation_metrics_kFold(kFoldExperiment, fold, best_metrics, best_figures)
    log_kFold_average(kFoldExperiment, results)



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





def trainModel(trainloader, testloader, net, optimizer, device, parameters, val_ratio, train_ratio):

    # Enable tensorbard logging
    tb, experiment = CreateLoggers(parameters, net)

    ###### The training loop
    t0 = time.time()
    max_batch = 0


    best_eval_score, best_eval_epoch = 0, 0
    best_eval_metrics, best_eval_figures = None, None
    for epoch in range(parameters["epochs"]): 

        # Evaluate validation performance
        metrics, figures = evalValidation(testloader, net, device, val_ratio, tb, experiment, epoch, parameters)

        if metrics[parameters["earlyStoppingMetric"]] > best_eval_score:
                best_eval_score = metrics[parameters["earlyStoppingMetric"]]
                best_eval_epoch = epoch
                best_eval_metrics = metrics
                best_eval_figures = figures

        if parameters["earlyStopping"]:
            if epoch - best_eval_epoch >= parameters["earlyStoppingPatience"]:
                print(f"Early stopping applied (best metric={best_eval_score})")
                break


        # Actual training happens here
        with experiment.train():
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
                    tb.add_scalar("Training Loss", running_loss/(max_batch+1), epoch)
                    running_loss = 0.0
            max_batch = i

    print('Finished Training')
    t1 = time.time()
    save_model(net)
    
    print(f"Total time taken: {t1 - t0}")
    return best_eval_metrics, best_eval_figures



def evalValidation(testloader, net, device, val_ratio, tb, experiment, epoch, parameters):
    """Perform evaluation on validation set and log using tensorboard"""
    net.eval()
    with experiment.validate():
        with torch.no_grad():
            y_pred, y_output, y_true = None, None, None
            weights_all_batches = None
            for data in testloader:
                features, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)

                y_output_batch = net(features)
                y_pred_batch = (y_output_batch > 0.5).int()

                useSampleWeighting = (parameters["data_sample_mode"] == "weighted")

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
            y_true, y_pred, y_output = y_true.cpu().numpy(), y_pred.cpu().numpy(), y_output.cpu().numpy()

            eval_metrics, eval_figures = get_evaluation_metrics(y_true, y_output, y_pred)
            eval_metrics["Validation Loss"] = loss.detach().item()
            log_evaluation_metrics(tb, experiment, epoch, eval_metrics, eval_figures)

            return eval_metrics, eval_figures


def save_model(net):
    if net.embeddingType == "protBert":
        net.firstLayer = None
    torch.save(net.state_dict(), "model.txt")

def load_model(filestring):
    net = torch.save(filestring)
    if net.embeddingType == "protBert":
        net.firstLayer = firstLayer(net.device, net.embeddingType)
    return net
    

if __name__ == "__main__":

    parameters = { 
        "gpu_mode": True,
        "epochs": 100,
        "batch_size": 2048,
        "learning_rate": 0.003,
        "redundancyPercentage": 50,
        "aminoAcid": "Hydroxylation-K",
        "embeddingType": "embeddingLayer",
        "test_data_ratio": 0.1,
        "data_sample_mode": "oversample",
        "weight_decay": 1,
        "model": FCNet,
        "loss_function": nn.BCEWithLogitsLoss,
        "optimizer": optim.AdamW,
        "crossValidation": True,
        "folds": 5,
        "earlyStopping": True,
        "earlyStoppingMetric": "AUC ROC",
        "earlyStoppingPatience": 10,
        }                             

    testModel(parameters)
    exit()
    AAs = ["Acetylation", "Hydroxylation-K", "Hydroxylation-P", "Methylation-K", "Methylation-R", "N-linked Glycosylation", "O-linked Glycosylation", "Phosphorylation-['S', 'T']", "Phosphorylation-Y", "Pyrrolidone carboxylic acid", "S-palmitoylation-C", "Sumoylation", "Ubiquitination"]
    for aa in AAs:
        parameters["aminoAcid"] = aa
        testModel(parameters)
