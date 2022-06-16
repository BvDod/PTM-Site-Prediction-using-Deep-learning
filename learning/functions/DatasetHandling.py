import comet_ml
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def indexListToOneHot(input_array):
    """This function converts an array of categorical features to an array of one-hot represented features"""

    OneHotVariables = 27
    samples, columns = input_array.shape

    output_array = np.zeros((samples, OneHotVariables*columns))
    for i in range(samples):
        for j in range(columns):
            output_array[i, (j*OneHotVariables) + input_array[i,j]] = 1
    return output_array


def get_folder_name(AA, embeddingType, test=False, MusiteTest=False, split_2010=False):
    """Get folder name of AA of particular redundancy level"""

    embeddingToFolder = {
        "oneHot":  "onehot",
        "embeddingLayer": "indices",
        "adaptiveEmbedding": "indices",
        "protBert": "embeddings",
    }

    type_folder = embeddingToFolder[embeddingType]

    data_dir = "code/Thesis/dataset/"
    if not test:
        if split_2010:
            folder_name = f"{data_dir}train_2010/{AA}/{type_folder}"
            folder_species = f"{data_dir}train_2010/{AA}"
        else:
            folder_name = f"{data_dir}train/{AA}/{type_folder}"
            folder_species = f"{data_dir}train/{AA}"
    if test:
        if split_2010:
            folder_name = f"{data_dir}test_2010/{AA}/{type_folder}"
            folder_species = f"{data_dir}test_2010/{AA}"
        else:
            folder_name = f"{data_dir}test/{AA}/{type_folder}"
            folder_species = f"{data_dir}test/{AA}"
    elif MusiteTest:
        folder_name = f"{data_dir}test_Musiteown/{AA}/{type_folder}"
        folder_species = f"{data_dir}test_Musiteown/{AA}"
    return folder_name, folder_species


def split_training_test_data(X, y, parameters, fold, tensor_dtype=torch.float):
    """ Splits the training and test data into fractions NOTE: NO SHUFFLING, ALREADY DID THAT"""
    
    CV = parameters["crossValidation"]
    test_data_ratio = parameters["test_data_ratio"]
    k = parameters["folds"]

    if CV:
        kfold = KFold(n_splits=k, shuffle=True, random_state=parameters["random_state"])
        train_ids, val_ids = list(kfold.split(X))[fold]
        X_train, y_train = X[train_ids, :], y[train_ids, :]
        X_val, y_val = X[val_ids, :], y[val_ids, :]

    else:
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=test_data_ratio, random_state=parameters["random_state"])


    X_val = torch.tensor(X_val, dtype=tensor_dtype)
    if parameters["predictSpecies"] and not parameters["onlyPredictHumans"]:
        y_val = torch.tensor(y_val, dtype=torch.long)
    else:
        y_val = torch.tensor(y_val, dtype=torch.float)
    X_train = torch.tensor(X_train, dtype=tensor_dtype)
    if parameters["predictSpecies"] and not parameters["onlyPredictHumans"]:
        y_train = torch.tensor(y_train, dtype=torch.long)
    else:
        y_train = torch.tensor(y_train, dtype=torch.float)

    return X_train, y_train, X_val, y_val


def loadData(parameters, aminoAcid, test=False, MusiteTest=False, split_2010= False):
    """Function used to load correct dataset based on parameters used"""

    asOneHot = parameters["embeddingType"] == "oneHot"
    tensor_dtype = torch.float if ((parameters["embeddingType"] == "oneHot") or (parameters["embeddingType"] == "protBert")) else torch.int

    folder, folder_species = get_folder_name(aminoAcid, parameters["embeddingType"], test=test, MusiteTest=MusiteTest, split_2010=split_2010)
    X_neg, y_neg = np.load(f"{folder}/X_train_neg.npy"), np.load(f"{folder}/y_train_neg.npy")
    X_pos, y_pos = np.load(f"{folder}/X_train_pos.npy"), np.load(f"{folder}/y_train_pos.npy")

    species_pos, species_neg = np.load(f"{folder_species}/species_pos.npy").astype(int), np.load(f"{folder_species}/species_neg.npy").astype(int)
    
    if parameters["onlyPredictHumans"]:
        species_pos, species_neg = (species_pos == 0).astype(int), (species_neg == 0).astype(int)
    
    n = len(y_neg) + len(y_pos)

    y_pos = np.vstack((y_pos, species_pos)).T
    y_neg = np.vstack((y_neg, species_neg)).T

    n = len(y_neg) + len(y_pos)

    return X_neg, y_neg, X_pos, y_pos, n, tensor_dtype


def createDatasets(X_train_neg, y_train_neg, X_val_neg, y_val_neg, X_train_pos, y_train_pos, X_val_pos, y_val_pos, reduceNegativeSamples=False):
    """Function used to create datasets"""

    n_train_neg, n_train_pos = X_train_neg.shape[0], X_train_pos.shape[0]
    n_val_neg, n_val_pos = X_val_neg.shape[0], X_val_pos.shape[0]

    train_ratio, val_ratio = n_train_neg/n_train_pos, n_val_neg/n_val_pos

    train_weight = torch.tensor(np.concatenate([np.full(n_train_pos, 1), np.full(n_train_neg, 1./train_ratio)]))
    val_weight = torch.tensor(np.concatenate([np.full(n_val_pos, 1), np.full(n_val_neg, 1./val_ratio)]))

    if reduceNegativeSamples == True:
        X_train_neg = X_train_neg[:n_train_pos,:]
        y_train_neg = y_train_neg[:n_train_pos,:]
    
    
    X_train = torch.cat([X_train_pos, X_train_neg],dim=0)
    del X_train_pos
    del X_train_neg
    y_train = torch.cat([y_train_pos, y_train_neg],dim=0)
    del y_train_pos
    del y_train_neg
    X_val = torch.cat([X_val_pos, X_val_neg],dim=0)
    del X_val_pos
    del X_val_neg
    y_val = torch.cat([y_val_pos, y_val_neg],dim=0)
    del y_val_pos
    del y_val_neg

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_val, y_val) 
    return trainset, testset, train_weight, val_weight, n_train_pos, n_train_neg, train_ratio, val_ratio


def createDataset(X_train_neg, y_train_neg, X_train_pos, y_train_pos, reduceNegativeSamples=False):
    """Function used to create datasets"""

    n_train_neg, n_train_pos = X_train_neg.shape[0], X_train_pos.shape[0]
    train_ratio = n_train_neg/n_train_pos 

    train_weight = torch.tensor(np.concatenate([np.full(n_train_pos, 1), np.full(n_train_neg, 1./train_ratio)]))

    if reduceNegativeSamples == True:
        X_train_neg = X_train_neg[:n_train_pos,:]
        y_train_neg = y_train_neg[:n_train_pos,:]
    
    X_train = torch.cat([X_train_pos, X_train_neg],dim=0)
    del X_train_pos
    del X_train_neg
    y_train = torch.cat([y_train_pos, y_train_neg],dim=0)
    del y_train_pos
    del y_train_neg

    trainset = TensorDataset(X_train, y_train)

    return trainset, train_weight, n_train_pos, n_train_neg, train_ratio


def CreateDataloaders(trainset, testset, n_train_pos, n_train_neg, parameters, train_weight, data_sample_mode, dataloader_samples, batch_size):
    """ Create dataloaders off training and test-set based on type of sampling technique used """
    if not data_sample_mode in ["undersample", "oversample", "weighted", "balanced", "unbalanced", "focalLoss"]:
        print("Error: invalid sampling method ")
        exit()

    if (data_sample_mode == "undersample") or (data_sample_mode == "oversample"):

        if data_sample_mode == "undersample":
            sampler_train = torch.utils.data.sampler.WeightedRandomSampler(train_weight,dataloader_samples, replacement= (len(parameters["aminoAcid"]) > 1))
            
        elif data_sample_mode == "oversample":
            sampler_train = torch.utils.data.sampler.WeightedRandomSampler(train_weight,dataloader_samples, replacement=True)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train, pin_memory=False)

    else:
        if len(parameters["aminoAcid"]) > 1:
            sampler = torch.utils.data.sampler.RandomSampler(trainset, num_samples = dataloader_samples, replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, sampler=sampler)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return trainloader, testloader


def CreateDataloader(testset, parameters, batch_size):
    """ Create dataloaders off training and test-set based on type of sampling technique used """

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return testloader