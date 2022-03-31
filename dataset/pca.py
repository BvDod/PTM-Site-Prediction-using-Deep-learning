# %%
import os
from random import sample
from cv2 import transform
import pandas as pd
import string
from transformers import BertTokenizer
import re

import sys
sys.path.insert(0,'C:\\Users\\bdode\\Documents\\msc-thesis\\Thesis\\')
os.chdir("C:\\Users\\bdode\\Documents\\msc-thesis\\Thesis\\")

import matplotlib.pyplot as plt
import numpy as np

from learning.models.firstLayer import protBertEmbedding
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = protBertEmbedding(device)

source_dir = "dataset/data/processed/yearsAdded/"
outdir = "dataset/data/motifs/"

df_uniprot_animals = pd.read_csv("dataset/data/uniprot_animals.csv")

# Get all full UniProt sequences for all dbPTM source files
# Get all full UniProt sequences for all dbPTM source files

transformations = {
    "Embedding of site": lambda x: x[:,15,],
    "Mean": lambda x: np.mean(x, axis=1),
    "Max": lambda x: np.amax(x, axis=1)
}

fig = plt.figure(constrained_layout=True, figsize=(15,80))
fig.suptitle("t-SNE", fontsize=35)
subfigs = fig.subfigures(len(os.listdir(source_dir))//2, 1)

file_i = -1
for file in os.listdir(source_dir):

    # Check if already processed, else skip
    if file.split("_")[-3] == "neg":
        continue
    file_i += 1

    subfig = subfigs[file_i]
    subfig.suptitle(file.split('_')[0], fontsize=25)

    file_neg = file.replace('pos', 'neg')

    df_pos = pd.read_csv(f"{source_dir}/{file}", delimiter=",")
    df_neg = pd.read_csv(f"{source_dir}/{file_neg}", delimiter=",")

    df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence

    dfs = {}

    """
    dfs = {
        "All Sites - Pos.": df_pos,
        "All Sites - Neg.": df_neg,
    }
    
    if file.split("_")[0] == "O-linked Glycosylation" or file.split("_")[0] == "Phosphorylation-['S', 'T']":
        dfs["S - positive"] = df_pos[df_pos["ModifiedAA"] == "S"]
        dfs["S - negative"] = df_neg[df_neg["ModifiedAA"] == "S"]
        dfs["T - positive"] = df_pos[df_pos["ModifiedAA"] == "T"]
        dfs["T - negative"] = df_neg[df_neg["ModifiedAA"] == "T"]
    """
    
    df_human_pos = df_pos[df_pos['UniprotID'].str.split('_').str[-1] == "HUMAN"]
    df_human_neg = df_neg[df_neg['UniprotID'].str.split('_').str[-1] == "HUMAN"]
    dfs["Human - Pos."] = df_human_pos
    dfs["Human - Neg."] = df_human_neg

    df_animals_pos = df_pos[df_pos['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    df_animals_neg = df_neg[df_neg['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    dfs["Animals - Pos."] = df_animals_pos
    dfs["Animals - Neg."] = df_animals_neg

    df_non_animals_pos = df_pos[~df_pos['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    df_non_animals_neg = df_neg[~df_neg['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    dfs["Non-Animals - Pos."] = df_non_animals_pos
    dfs["Non-Animals - Neg."] = df_non_animals_neg
       
    sample_array = None
    labels = []
    for i, (label, df) in enumerate(dfs.items()):
        n = 2500//len(dfs)
        if len(df) < n:
            n = len(df)
        seqs = df["TruncatedUniProtSequence"].sample(n).tolist()
        seqs = [" ".join(seq) for seq in seqs]
        seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True, max_length=33, truncation=True)
        ids = torch.tensor(np.array(ids['input_ids']), device=device)
        embeddings = model(ids)
        
        if sample_array is None:
            sample_array = embeddings.cpu().numpy()
        else:
            sample_array = np.concatenate([sample_array, embeddings.cpu().numpy()], axis=0)
        labels = labels + [i]*n
    

    axes = subfig.subplots(nrows=1, ncols=3)

    for trans_j, (name, transform) in enumerate(transformations.items()):
        sample_array_transformed = transform(sample_array)

        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        X = pca.fit_transform(sample_array_transformed)

        ax = axes[trans_j]
        plot = ax.scatter(X[:,0], X[:,1], c=labels, s=10, alpha=1)
        ax.legend(handles=plot.legend_elements()[0], labels=list([key for key in dfs.keys()]))
        ax.set_title(name)
plt.savefig(f"dataset/data/pca_species_more.PNG", dpi=500, bbox_inches='tight')

# %%



# %%
        exit()

        
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].set_title(label, fontsize=7)
        ax[i].axis('off')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"dataset/data/motifs/{file.split('_')[0]}.PNG", dpi=500)
    
        
    