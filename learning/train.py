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
    learning_rate = 0.005
    redundancyPercentage = 50
    # aminoAcid = "Phosphorylation-['S', 'T']"
    # aminoAcid = "Hydroxylation-K"
    aminoAcid = "O-linked Glycosylation"
    test_data_ratio = 0.1

    data_sample_mode = "undersample"
    #data_sample_mode = "weighted"
    #data_sample_mode = "oversample"
    #data_sample_mode = "balanced"

    model = EmbFC_Net
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

    if isinstance(net, FC_Net):
        tensor_dtype = torch.float
    else:
        tensor_dtype = torch.int

    X_train_neg, y_train_neg, X_val_neg, y_val_neg = split_training_test_data(X_neg,y_neg, test_data_ratio=test_data_ratio, tensor_dtype=tensor_dtype)
    X_train_pos, y_train_pos, X_val_pos, y_val_pos = split_training_test_data(X_pos,y_pos, test_data_ratio=test_data_ratio, tensor_dtype=tensor_dtype)

    n_train_neg, n_train_pos = X_train_neg.shape[0], X_train_pos.shape[0]
    n_val_neg, n_val_pos = X_val_neg.shape[0], X_val_pos.shape[0]
    print(n_train_neg, n_train_pos)
    train_ratio = n_train_neg/n_train_pos
    val_ratio = n_val_neg/n_val_pos

    train_weight = torch.tensor(np.concatenate([np.full(n_train_pos, 1), np.full(n_train_neg, 1./train_ratio)]))
    val_weight = torch.tensor(np.concatenate([np.full(n_val_pos, 1), np.full(n_val_neg, 1./val_ratio)]))
    print(f"Loaded folder {folder} ({n} samples)")

    if data_sample_mode == "balanced":
        X_train_neg = X_train_neg[:n_train_pos,:]
        y_train_neg = y_train_neg[:n_train_pos]
    X_val_neg = X_val_neg[:n_val_pos,:]
    y_val_neg = y_val_neg[:n_val_pos]
 


    X_train = torch.cat([X_train_pos, X_train_neg],dim=0)
    y_train = torch.cat([y_train_pos, y_train_neg],dim=0)
    X_val = torch.cat([X_val_pos, X_val_neg],dim=0)
    y_val = torch.cat([y_val_pos, y_val_neg],dim=0)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_val, y_val) 

    if (data_sample_mode == "undersample") or (data_sample_mode == "oversample"):
        if data_sample_mode == "undersample":
            sampler_train = torch.utils.data.sampler.WeightedRandomSampler(train_weight, n_train_pos*2)
            
        elif data_sample_mode == "oversample":
            sampler_train = torch.utils.data.sampler.WeightedRandomSampler(train_weight,n_train_neg*2, replacement=True)
            
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,)

    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    weight_decay = 0.0100003
    optimizer = optimizer(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if model == EmbFC_Net:
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.embeddingSize}/{net.FC_layer_sizes}/{data_sample_mode}/{weight_decay}/"
    else:
        log_dir = f"runs/{aminoAcid}/{net.model_name}/red_{redundancyPercentage}/{net.FC_layer_sizes}/{data_sample_mode}/{weight_decay}/"


    tb = SummaryWriter(log_dir=log_dir)
    t0 = time.time()

    # The training loop
    max_batch = None
    for epoch in range(epochs): 
        
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

            if i == max_batch:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / eps_per_loss))
                tb.add_scalar("Training Loss", running_loss/eps_per_loss, epoch)
                running_loss = 0.0
                correct = 0
        max_batch = i

        
    print('Finished Training')
    t1 = time.time()
    print(f"Total time taken: {t1 - t0}")
