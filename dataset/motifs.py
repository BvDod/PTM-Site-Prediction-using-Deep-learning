from Bio import motifs
import os
import pandas as pd
from Bio.Seq import Seq
import string
from PIL import Image

import matplotlib.pyplot as plt


source_dir = "dataset/data/processed/yearsAdded/"
outdir = "dataset/data/motifs/"

df_uniprot_animals = pd.read_csv("dataset/data/uniprot_animals.csv")

# Get all full UniProt sequences for all dbPTM source files
# Get all full UniProt sequences for all dbPTM source files
for file in os.listdir(source_dir):
    # Check if already processed, else skip
    if file.split("_")[-3] == "neg":
        continue

    file_neg = file.replace('pos', 'neg')

    df_pos = pd.read_csv(f"{source_dir}/{file}", delimiter=",")
    df_neg = pd.read_csv(f"{source_dir}/{file_neg}", delimiter=",")

    df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence

    dfs = {
        "All Sites - positive": df_pos,
        "All Sites - negative": df_neg,
    }
    
    if file.split("_")[0] == "O-linked Glycosylation" or file.split("_")[0] == "Phosphorylation-['S', 'T']":
        dfs["S - positive"] = df_pos[df_pos["ModifiedAA"] == "S"]
        dfs["S - negative"] = df_neg[df_neg["ModifiedAA"] == "S"]
        dfs["T - positive"] = df_pos[df_pos["ModifiedAA"] == "T"]
        dfs["T - negative"] = df_neg[df_neg["ModifiedAA"] == "T"]


    df_human_pos = df_pos[df_pos['UniprotID'].str.split('_').str[-1] == "HUMAN"]
    df_human_neg = df_neg[df_neg['UniprotID'].str.split('_').str[-1] == "HUMAN"]
    dfs["Human only - positive"] = df_human_pos
    dfs["Human only - negative"] = df_human_neg

    df_animals_pos = df_pos[df_pos['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    df_animals_neg = df_neg[df_neg['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    dfs["Animals only - positive"] = df_animals_pos
    dfs["Animals only - negative"] = df_animals_neg

    df_non_animals_pos = df_pos[~df_pos['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    df_non_animals_neg = df_neg[~df_neg['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
    dfs["Non-Animals only - positive"] = df_non_animals_pos
    dfs["Non-Animals only - negative"] = df_non_animals_neg
    

    fig, ax = plt.subplots(int(len(dfs)/2), 2,)
    print(file.split("_")[0])
    ax = list(ax.flatten())
    fig.suptitle(file.split("_")[0], fontsize=18)
    for i, (label, df) in enumerate(dfs.items()):
        seqs = df["TruncatedUniProtSequence"].tolist()
        motif = motifs.create([Seq(sequence) for sequence in seqs], alphabet=string.ascii_uppercase + "-")
        motif.weblogo(f"{outdir}/temp.PNG", format = "png_print", alphabet="alphabet_protein")
        img = Image.open(f"{outdir}/temp.PNG")
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img)
        ax[i].set_title(label, fontsize=7)
        ax[i].axis('off')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"dataset/data/motifs/{file.split('_')[0]}.PNG", dpi=500)
    
        
    