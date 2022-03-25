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

        # Create the different layers
        self.layers = nn.ModuleList()
        self.layers.append(firstLayer(device, embeddingType=parameters["embeddingType"], embeddingSize=parameters["embeddingSize"], embeddingDropout=parameters["embeddingDropout"]))

        for i in range(parameters["CNN_layers"]):
            self.layers.append(CNN_Layer(self.layers[-1], parameters))
        
        if parameters["LSTM_layers"] > 0:
            self.layers.append(LSTM_Layer(self.layers[-1], parameters))

        self.layers.append(BahdanauAttention(self.layers[-1], in_features=self.layers[-1].outputDimensions, hidden_units=10,num_task=1))

        layer_size = parameters["FC_layer_size"]
        for i in range(parameters["FC_layers"]):
            self.layers.append(FC_Layer(self.layers[-1], layer_size, dropoutPercentage=parameters["FC_dropout"]))
            layer_size = 32
            parameters["FC_dropout"] = 0

        self.layers.append(FC_Layer(self.layers[-1], 1, useActivation=False, dropoutPercentage=0))



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x