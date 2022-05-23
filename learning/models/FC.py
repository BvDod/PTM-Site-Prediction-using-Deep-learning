import torch

import torch.nn as nn
import torch.nn.functional as F
from .firstLayer import firstLayer

from .layerClasses import FC_Layer, CNN_Layer, LSTM_Layer, BahdanauAttention



class FCNet(nn.Module):
    """Neural net which only uses fully connected layers, uses one-hot encoded features as input"""

    def __init__(self, device, parameters, peptide_size=33):
        super().__init__()
        self.model_parameters = parameters
        self.device = device

        if not "layerToSplitOn" in parameters:
            parameters["layerToSplitOn"] = "FC"

        # Create the different layers
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        if parameters["CNNType"] == "Musite":
            if parameters["embeddingType"] == "oneHot":
                self.layers.append(firstLayer(device, embeddingType=parameters["embeddingType"], embeddingSize=None, embeddingDropout=0))
                self.layers.append(CNN_Layer(self.layers[-1], parameters, dropoutPercentage=0.75, filters=200, kernel_size=1))
            else:
                if (self.model_parameters["useSpecieFeature"] == True) and (self.model_parameters["SpecieFeatureHoldout"] == False):
                    self.layers.append(firstLayer(device, embeddingType=parameters["embeddingType"], embeddingSize=200, embeddingDropout=0.75, layerNorm=False, add_extra=21))
                else:
                    self.layers.append(firstLayer(device, embeddingType=parameters["embeddingType"], embeddingSize=200, embeddingDropout=0.75, layerNorm=False, add_extra=0))
            
            
            self.layers.append(CNN_Layer(self.layers[-1], parameters, dropoutPercentage=0.75, filters=150, kernel_size=9,))
            self.layers.append(CNN_Layer(self.layers[-1], parameters,dropoutPercentage= 0.75, filters=200, kernel_size=10, maxPool=True))

        elif parameters["CNNType"] == "Adapt":
            self.layers.append(firstLayer(device, embeddingType=parameters["embeddingType"], embeddingSize=32, embeddingDropout=0))
            self.layers.append(CNN_Layer(self.layers[-1], parameters, dropoutPercentage=0, filters=256, kernel_size=10,batchNorm=True, maxPool=True))

        else:
            print("Error: invalid CNN-type string")
            exit()

        
        if parameters["LSTM_layers"] > 0:
            if not parameters["layerToSplitOn"] == "LSTM":
                self.layers.append(LSTM_Layer(self.layers[-1], parameters))
        
        if (not parameters["layerToSplitOn"] == "LSTM") and (not parameters["layerToSplitOn"] == "transformer") and (self.model_parameters["SpecieFeatureHoldout"] == True):
            self.layers.append(BahdanauAttention(self.layers[-1], in_features=self.layers[-1].outputDimensions, hidden_units=10,num_task=1, add_extra=21))
        else:
            self.layers.append(BahdanauAttention(self.layers[-1], in_features=self.layers[-1].outputDimensions, hidden_units=10,num_task=1))


        """
        layer_size = parameters["FC_layer_size"]
        for i in range(parameters["FC_layers"]):
            self.layers.append(FC_Layer(self.layers[-1], layer_size, dropoutPercentage=parameters["FC_dropout"]))
            layer_size = 32
            parameters["FC_dropout"] = 0
        """
        if (len(parameters["aminoAcid"]) > 1) and (not parameters["MultiTask_Species"]):
            for task in range(len(parameters["aminoAcid"])):
                headLayers = nn.ModuleList()

                if (parameters["layerToSplitOn"] == "LSTM"):
                    headLayers.append(LSTM_Layer(self.layers[-1], parameters))
                    headLayers.append(BahdanauAttention(headLayers[-1], in_features=headLayers[-1].outputDimensions, hidden_units=10,num_task=1))
                
                if (parameters["layerToSplitOn"] == "transformer"):
                    headLayers.append(BahdanauAttention(self.layers[-1], in_features=self.layers[-1].outputDimensions, hidden_units=10,num_task=1))

                if (parameters["layerToSplitOn"] == "FC"):
                    if parameters["FCType"] == "Adapt":
                        headLayers.append(FC_Layer(self.layers[-1], 32, dropoutPercentage=0.5))
                        headLayers.append(FC_Layer(headLayers[-1], 1, useActivation=False, dropoutPercentage=0))
                    
                    if parameters["FCType"] == "Musite":
                        headLayers.append(FC_Layer(self.layers[-1], 149, dropoutPercentage=0.2982))
                        headLayers.append(FC_Layer(headLayers[-1], 8, dropoutPercentage=0))
                        headLayers.append(FC_Layer(headLayers[-1], 1, useActivation=False, dropoutPercentage=0))
                else:
                    if parameters["FCType"] == "Adapt":
                        headLayers.append(FC_Layer(headLayers[-1], 32, dropoutPercentage=0.5))
                        headLayers.append(FC_Layer(headLayers[-1], 1, useActivation=False, dropoutPercentage=0))
                    
                    if parameters["FCType"] == "Musite":
                        headLayers.append(FC_Layer(headLayers[-1], 149, dropoutPercentage=0.2982))
                        headLayers.append(FC_Layer(headLayers[-1], 8, dropoutPercentage=0))
                        headLayers.append(FC_Layer(headLayers[-1], 1, useActivation=False, dropoutPercentage=0))


                headLayers.append(nn.Sigmoid())
                self.heads.append(headLayers)

        elif parameters["MultiTask_Species"]:
            headLayers = nn.ModuleList()

            headLayers.append(FC_Layer(self.layers[-1], 32, dropoutPercentage=0.5))
            headLayers.append(FC_Layer(headLayers[-1], 1, useActivation=False, dropoutPercentage=0))
            headLayers.append(nn.Sigmoid())
            self.heads.append(headLayers)

            headLayers2 = nn.ModuleList()

            headLayers2.append(FC_Layer(self.layers[-1], 32, dropoutPercentage=0.5))
            headLayers2.append(FC_Layer(headLayers2[-1], 21, useActivation=False, dropoutPercentage=0))
            headLayers2.append(nn.Softmax(dim=1))
            self.heads.append(headLayers2)

        else:
            if parameters["FCType"] == "Adapt":
                if parameters["useSpecieFeature"] == True:
                    self.layers.append(FC_Layer(self.layers[-1], 32, dropoutPercentage=0.5))
                else:
                    self.layers.append(FC_Layer(self.layers[-1], 32, dropoutPercentage=0.5))

                if parameters["predictSpecies"] and not parameters["onlyPredictHumans"]:
                    self.layers.append(FC_Layer(self.layers[-1], 21, useActivation=False, dropoutPercentage=0))
                else:
                    self.layers.append(FC_Layer(self.layers[-1], 1, useActivation=False, dropoutPercentage=0))
                    
            if parameters["FCType"] == "Musite":
                self.layers.append(FC_Layer(self.layers[-1], 149, dropoutPercentage=0.2982))
                self.layers.append(FC_Layer(self.layers[-1], 8, dropoutPercentage=0))
                if parameters["predictSpecies"] and not parameters["onlyPredictHumans"]:
                    self.layers.append(FC_Layer(self.layers[-1], 1, useActivation=False, dropoutPercentage=0))

            if parameters["predictSpecies"] and not parameters["onlyPredictHumans"]:
                self.layers.append(nn.Softmax(dim=1))
            else:
                self.layers.append(nn.Sigmoid())


    def forward(self, x, task_indexes, species):

        for layer in self.layers:
            x = layer(x)
            
            
            if (self.model_parameters["useSpecieFeature"] == True) and isinstance(layer, BahdanauAttention):
                if self.model_parameters["SpecieFeatureHoldout"] == True:
                    x = torch.cat((x, nn.functional.one_hot(species.long(), num_classes=21)), dim=1)
            
            if (self.model_parameters["useSpecieFeature"] == True) and isinstance(layer, firstLayer):
                if self.model_parameters["SpecieFeatureHoldout"] == False:
                    species = nn.functional.one_hot(species.long(), num_classes=21)
                    species = species.unsqueeze(1).repeat(1, 33, 1)
                    x = torch.cat((x, species), dim=2)


        if self.model_parameters["MultiTask_Species"]:
            output = []
        
            x_head = x[task_indexes[0]:task_indexes[1],:]
            for layer in self.heads[0]:
                if type(layer) == BahdanauAttention:
                    x_head = layer(x_head, task_indexes=[task_indexes[0],task_indexes[1]])
                else:
                    x_head = layer(x_head)
            output.append(x_head)

            x_head = x[task_indexes[0]:task_indexes[1],:]
            for layer in self.heads[1]:
                if type(layer) == BahdanauAttention:
                    x_head = layer(x_head, task_indexes=[task_indexes[0],task_indexes[1]])
                else:
                    x_head = layer(x_head)
            output.append(x_head)

            output = torch.cat(output, axis=1)
            return output

        elif len(self.heads) > 0:
            output = []
            for i in range(len(task_indexes[:-1])):
                x_head = x[task_indexes[i]:task_indexes[i+1],:]
                for layer in self.heads[i]:
                    if type(layer) == BahdanauAttention:
                        x_head = layer(x_head, task_indexes=[task_indexes[i],task_indexes[i+1]])
                    else:
                        x_head = layer(x_head)
                output.append(x_head)
            output = torch.cat(output, axis=0)
            return output
        else:
            return x