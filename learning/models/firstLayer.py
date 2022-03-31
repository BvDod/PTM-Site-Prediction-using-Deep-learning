import torch.nn as nn
import torch 
from transformers import BertModel

class firstLayer(nn.Module):
    """ Used as the first layer of other neural networks, can use different types of representation (one-hot, embedding, adaptive-embedding, bertEmbeddings) """

    def __init__(self, device, embeddingType, embeddingSize=25, peptideSize=33, embeddingDropout=0, layerNorm=True):
        super().__init__()
        self.device =  device
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
        self.rolled_out = False


        self.layers = nn.ModuleList()

        ### Create model layers based on type of embedding
        if embeddingType == "oneHot":
            self.layers.append(oneHot())
            self.outputDimensions = 27
            
        elif embeddingType == "embeddingLayer":
            self.layers.append(nn.Embedding(27, self.embeddingSize))
            self.layers.append(nn.Dropout(embeddingDropout))
            self.outputDimensions = embeddingSize
        
        elif embeddingType == "adaptiveEmbedding":
            self.layers.append(adaptiveEmbedding(device, embeddingSize=self.embeddingSize, layerNorm=layerNorm))
            self.layers.append(nn.Dropout(embeddingDropout))
            self.outputDimensions = embeddingSize
        

        elif embeddingType == "protBert":
            self.layers.append(protBertEmbedding(device))
            self.layers.append(nn.Dropout(embeddingDropout))
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
    def __init__(self, device, embeddingSize=5, peptideSize=33, layerNorm=True):
        super().__init__()
        self.device = device
        
        self.AAEmbedding = nn.Embedding(27, embeddingSize)
        self.positionEmbedding = nn.Embedding(peptideSize, embeddingSize)
        self.norm = nn.LayerNorm(embeddingSize)

    def forward(self, x):
        positionIndex = torch.arange(33, device=self.device, dtype=torch.long)
        positionIndex = positionIndex.unsqueeze(0).expand_as(x)
        embedding = self.AAEmbedding(x) + self.positionEmbedding(positionIndex)
        embedding = self.norm(embedding)
        return embedding


class protBertEmbedding(nn.Module):
    def __init__(self, device, peptideSize=33):
        super().__init__()
        self.device = device
        self.peptideSize = peptideSize

        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").half()
        self.model = self.model.to(device)
        self.model = self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            embedding = self.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        return embedding.float()

class oneHot(nn.Module):
    def __init__(self, peptideSize=33):
        super().__init__()
        self.peptideSize = peptideSize
    
    def forward(self, x):
        og_shape = x.shape
        return torch.reshape(x, (og_shape[0], 33, (og_shape[1]//33)))
        


