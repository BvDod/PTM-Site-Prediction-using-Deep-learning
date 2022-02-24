import torch

import torch.nn as nn
import torch.nn.functional as F



class EmbFC_Net(nn.Module):
    """Neural net which using a single simple embedding layer with a FC NN attached, uses categorical index as input"""

    def __init__(self, peptide_size, FC_layer_sizes = [1,], embedding_size = 3):
        super().__init__()
        self.embeddingSize = embedding_size
        self.embedding = torch.nn.Embedding(27, self.embeddingSize)
        self.model_name = "EMB-FC"

        # Create all fully connected layers
        self.FC_layer_sizes = [self.embeddingSize*peptide_size] + FC_layer_sizes
        self.FClayers = nn.ModuleList()
        for i, size in enumerate(self.FC_layer_sizes[:-1]):
            self.FClayers.append(nn.Linear(size, self.FC_layer_sizes[i+1]))

        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)

        for layer in self.FClayers[:-1]:
            x = F.relu(layer(x))

        x = self.sig(self.FClayers[-1](x))
        return x