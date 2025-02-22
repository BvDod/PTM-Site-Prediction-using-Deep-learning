from Bio import motifs
import os
import pandas as pd
from Bio.Seq import Seq
import string
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import logomaker
import matplotlib.pyplot as plt




source_dir = "dataset/data/processed/yearsAdded1/"
outdir = "dataset/data/motifs"

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
    

    img_list = []
    labels = []
    full_df = pd.concat((list(dfs.items())[0][1], list(dfs.items())[1][1]))
    seqs = full_df["TruncatedUniProtSequence"].tolist()
    seqs = [seq[0:16] + "." + seq[17:]  for seq in seqs]
    df_count = logomaker.alignment_to_matrix(seqs)
    df_background = logomaker.transform_matrix(df_count, 
                                      from_type='counts', 
                                      to_type='probability')

    for i, (label, df) in enumerate(dfs.items()):
        plt.figure()
        seqs = df["TruncatedUniProtSequence"].tolist()
        seqs = [seq[0:16] + "." + seq[17:]  for seq in seqs]
        df_count = logomaker.alignment_to_matrix(seqs)
        for column in list(df_background.columns):
            if not column in list(df_count.columns):
                df_count[column] = 0.0
        
        df_count = df_count[df_background.columns]
        df_info = logomaker.transform_matrix(df_count, 
                                      from_type='counts', 
                                      to_type='information',
                                      background= df_background)
        # create Logo object
        ww_logo = logomaker.Logo(df_info,
                                font_name='Stencil Std',
                                color_scheme='NajafabadiEtAl2017',)
        plt.ylim(0, 0.1)
        plt.savefig(f"{outdir}/temp{i}.PNG")
        img = Image.open(f"{outdir}/temp{i}.PNG")
        img_list.append(img)
        labels.append(label)
    
    fig, ax = plt.subplots(int(len(dfs)/2), 2,)
    print(file.split("_")[0])
    ax = list(ax.flatten())
    fig.suptitle(file.split("_")[0], fontsize=18)
    for i, img in enumerate(img_list):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img)
        ax[i].set_title(labels[i], fontsize=7)
        ax[i].axis('off')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"dataset/data/motifs/{file.split('_')[0]}.PNG", dpi=500)
    
        
    