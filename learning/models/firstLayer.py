import torch.nn as nn
import torch 
from transformers import BertModel

class firstLayer(nn.Module):
    """ Used as the first layer of other neural networks, can use different types of representation (one-hot, embedding, adaptive-embedding, bertEmbeddings) """

    def __init__(self, device, embeddingType, embeddingSize=10, peptideSize=31):
        super().__init__()
        self.device =  device
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize

        self.layers = nn.ModuleList()

        ### Create model layers based on type of embedding
        if embeddingType == "oneHot":
            self.outputDimensions = 27

        elif embeddingType == "embeddingLayer":
            self.layers.append(nn.Embedding(27, self.embeddingSize))
            self.outputDimensions = embeddingSize

        elif embeddingType == "adaptiveEmbedding":
            self.layers.append(adaptiveEmbedding(device, embeddingSize=self.embeddingSize))
            self.outputDimensions = embeddingSize

        elif embeddingType == "protBert":
            self.layers.append(protBertEmbedding(device))
            self.outputDimensions = 1024

        else:
            print("Error: invalid embedding type")
            exit()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class adaptiveEmbedding(nn.Module):
    """ Adaptive embedding layer """
    def __init__(self, device, embeddingSize=5, peptideSize=31):
        super().__init__()
        self.device = device
        
        self.AAEmbedding = nn.Embedding(27, embeddingSize)
        self.positionEmbedding = nn.Embedding(peptideSize, embeddingSize)

    def forward(self, x):
        positionIndex = torch.arange(31, device=self.device, dtype=torch.long)
        positionIndex = positionIndex.unsqueeze(0).expand_as(x)
        embedding = self.AAEmbedding(x) + self.positionEmbedding(positionIndex)
        return embedding


class protBertEmbedding(nn.Module):
    def __init__(self, device, peptideSize=31):
        super().__init__()
        self.device = device
        self.peptideSize = peptideSize

        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").half()
        self.model = self.model.to(device)
        self.model = self.model.eval()
    
    def forward(self, x):
        with torch.no_grad():
            embedding = self.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        return embedding.float()

