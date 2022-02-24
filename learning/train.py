import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
import math

from models.FC import FC_Net
from models.EMB_FC import EmbFC_Net
from torch.utils.tensorboard import SummaryWriter


from functions.helper_functions import indexListToOneHot, get_folder_name, split_training_test_data, get_evaluation_metrics, log_evaluation_metrics




if __name__ == "__main__":

    ### Parameters
    gpu_mode = True
    epochs = 100
    batch_size = 16384
    learning_rate = 0.003
    redundancyPercentage = 70
    # aminoAcid = "Phosphorylation-['S', 'T']"
    # aminoAcid = "Hydroxylation-K"
    aminoAcid = "Phosphorylation-Y"
    test_data_ratio = 0.1

    data_sample_mode = "balanced"
    data_sample_mode = "weighted"
    data_sample_mode = "undersample"
    data_sample_mode = "oversample"

    model = FC_Net
    criterion = nn.BCELoss()
    optimizer = optim.AdamW

    ### Load training and test data
    folder = get_folder_name(aminoAcid, redundancyPercentage)
    X_neg, y_neg = np.load(f"{folder}/X_train_neg.npy"), np.load(f"{folder}/y_train_neg.npy")
    X_pos, y_pos = np.load(f"{folder}/X_train_pos.npy"), np.load(f"{folder}/y_train_pos.npy")

    n = len(y_neg) + len(y_pos)
    if model == FC_Net:
        X_neg = indexListToOneHot(X_neg)
        X_pos = indexListToOneHot(X_pos)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    net = model(peptide_size=31)
    net.to(device)

    X_train_neg, y_train_neg, X_val_neg, y_val_neg = split_training_test_data(X_neg,y_neg, test_data_ratio=test_data_ratio)
    X_train_neg, y_train_neg, X_val_neg, y_val_neg = split_training_test_data(X_neg,y_neg, test_data_ratio=test_data_ratio)

    print(X_train_neg.shape)
    print(X_train_pos.weight)
    exit()
    negative_sample_weight = len()


    if isinstance(net, FC_Net):
        X_val = torch.tensor(X_val, dtype=torch.float)
        X_train = torch.tensor(X_train, dtype=torch.float)
    else:
        X_val = torch.tensor(X_val, dtype=torch.int32)
        X_train = torch.tensor(X_train, dtype=torch.int32)
    print(f"Loaded folder {folder} ({n} samples)")

    y_train = torch.tensor(y_train, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_val, y_val) 

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    weight_decay = 0.01
    optimizer = optimizer(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if model == EmbFC_Net:
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.embeddingSize}/{net.FC_layer_sizes}/{weight_decay}/"
    else:
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.FC_layer_sizes}/{weight_decay}/"


    tb = SummaryWriter(log_dir=log_dir)
    t0 = time.time()

    # The training loop
    for epoch in range(epochs): 
        exit()
        
        # Evaluate validation performance
        if epoch % 1 == 0:
            y_pred, y_output, y_true = None, None, None
            with torch.no_grad():
                for data in testloader:
                    features, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)
                    y_output_batch = net(features).cpu().detach()
                    y_pred_batch = (y_output_batch > 0.5).int()
                    
                    if y_pred is None:
                        y_pred, y_output = y_pred_batch, y_output_batch
                        y_true = labels.cpu().detach()
                    else:
                        y_pred = torch.cat((y_pred, y_pred_batch), axis=0)
                        y_output = torch.cat((y_output, y_output_batch), axis=0)
                        y_true = torch.cat((y_true, labels.cpu().detach()), axis=0)

            loss = criterion(y_output, y_true)
            y_true, y_pred, y_output = y_true.numpy(), y_pred.numpy(), y_output.numpy()

            eval_metrics, eval_figures = get_evaluation_metrics(y_true, y_output, y_pred)
            log_evaluation_metrics(tb, epoch, eval_metrics, eval_figures)
            tb.add_scalar("Validation Loss", loss, epoch)
            
        
        # Actual training happens here
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].reshape((-1,1)).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print training loss
            batches = math.ceil((n*(1-test_data_ratio))/batch_size)
            eps_per_loss = batches
            running_loss += loss.item()

            if (i+1) % eps_per_loss == eps_per_loss-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / eps_per_loss))
                tb.add_scalar("Training Loss", running_loss/eps_per_loss, epoch)
                running_loss = 0.0
                correct = 0
                

        
    print('Finished Training')
    t1 = time.time()
    print(f"Total time taken: {t1 - t0}")
