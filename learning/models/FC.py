import torch

import torch.nn as nn
import torch.nn.functional as F
from .firstLayer import firstLayer

from .layerClasses import FC_Layer, CNN_Layer, LSTM_Layer



class FCNet(nn.Module):
    """Neural net which only uses fully connected layers, uses one-hot encoded features as input"""

    def __init__(self, device, peptide_size, embeddingType, FC_layer_sizes = [256,32,1]):
        super().__init__()
        self.embeddingType = embeddingType
        self.device = device
        self.model_name = "FC"
        self.FC_layer_sizes = FC_layer_sizes

        # Create all fully connected layers
        self.layers = nn.ModuleList()
        self.layers.append(firstLayer(device, embeddingType=embeddingType))
        self.layers.append(CNN_Layer(self.layers[-1]))
        self.layers.append(LSTM_Layer(self.layers[-1]))
        for size in FC_layer_sizes[:-1]:
            self.layers.append(FC_Layer(self.layers[-1], size))
        self.layers.append(FC_Layer(self.layers[-1], FC_layer_sizes[-1], useActivation=False))



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x