import torch

import torch.nn as nn
import torch.nn.functional as F



class FC_Net(nn.Module):
    """Neural net which only uses fully connected layers, uses one-hot encoded features as input"""

    def __init__(self, peptide_size, FC_layer_sizes = [10,1]):
        super().__init__()
        self.model_name = "FC"
        self.FClayers = nn.ModuleList()

        # Create all fully connected layers
        self.FC_layer_sizes = [27*peptide_size] + FC_layer_sizes
        for i, size in enumerate(self.FC_layer_sizes[:-1]):
            self.FClayers.append(nn.Linear(size, self.FC_layer_sizes[i+1]))

        self.sig = nn.Sigmoid()


    def forward(self, x):
        for layer in self.FClayers[:-1]:
            x = F.relu(layer(x))
        x = self.sig(self.FClayers[-1](x))
        return x