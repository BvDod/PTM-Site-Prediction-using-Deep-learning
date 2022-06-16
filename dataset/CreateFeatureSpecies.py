# %%
import os
import pandas as pd
import requests as r
import urllib
import io
import numpy as np
from transformers import BertTokenizer
import re


# %%
def indexListToOneHot(input_array):
    """This function converts an array of categorical features to an array of one-hot represented features"""

    OneHotVariables = 27
    samples, columns = input_array.shape

    output_array = np.zeros((samples, OneHotVariables*columns))
    for i in range(samples):
        for j in range(columns):
            output_array[i, (j*OneHotVariables) + input_array[i,j]] = 1
    return output_array

def get_merged_df(input_dir):
    dataframes = []
    for file in os.listdir(input_dir):

        df = pd.read_csv(f"{input_dir}/{file}", delimiter=",")

        if file.split("_")[-4] == "neg":
            df["label"] = 0
        elif file.split("_")[-4] == "pos":
            df["label"] = 1
            df = df.drop(columns=["truncateStatus","TruncatedUniProtSequence"])
        else:
            print("Error: invalid filename")
            exit()

        df["PTM_type"] = file.split("_")[0]
        dataframes.append(df)

    df_all = pd.concat(dataframes)
    return df_all

oneHotEncoding = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
            'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 
            'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 
            'Z': 25, "-": 26}


tokenizer = tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
# [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]



# For all retrieved full UniProt sequences, now truncate them
input_dir = "data/processed/final_split_if_2010/train"
df_all = get_merged_df(input_dir)

# %%
def get_species_names(df):
    df["Mnemonic"] = df.UniprotID.str.split("_").str[-1]
    df_mm = pd.read_csv(f"data/allmm.csv", delimiter="\t")

    df_mm.loc[df_mm["Common name"].isnull(), "Common name"] = df_mm.loc[df_mm["Common name"].isnull(), "Scientific name"]

    df = pd.merge(df, df_mm[["Mnemonic", "Common name"]], on="Mnemonic", how='left')
    df["Common name"] = df["Common name"].fillna("Unknown")
    df["Common name"] = df["Common name"].astype("string")
    return df

df_all = get_species_names(df_all)

# %%
species = df_all["Common name"].value_counts().head(20).index.tolist()
species.append("other")
indexes = range(len(species))

index_species = {
    "species": species,
    "label": indexes
}
df_index_species = pd.DataFrame(data=index_species)
df_index_species.to_csv("SpeciesLabels.csv", index=False)

input_dir = "data/processed/final_split_if_2010/train"
output_dir = "data/learningData/balanced/train_2010_noolddata/"

# %%

# %%
balance_dataset = False
for file in os.listdir(input_dir):
    if not file.split("_")[-4] == "pos":
        continue

    # check if already processed, else skip
    if os.path.exists(f"{output_dir}/{file}_oneHot"):
        continue

    print(file)
    
    import csv
    species_labels = pd.read_csv("SpeciesLabels.csv", index_col=0, header=0, squeeze=True).to_dict()



    df_pos = pd.read_csv(f"{input_dir}/{file}", delimiter=",")
    file_neg = file.replace('pos', 'neg')
    df_neg = pd.read_csv(f"{input_dir}/{file_neg}", delimiter=",")
    
    df_pos["y"] = 1
    df_neg["y"] = 0

    df_pos = get_species_names(df_pos)
    df_neg = get_species_names(df_neg)

    n = len(df_pos)

    column = "DateSeqModified"

    df_pos[column] = pd.to_datetime(df_pos[column])
    df_neg[column] = pd.to_datetime(df_neg[column])

    split_date = "2002"

    print(f"old sample count: {len(df_pos)+len(df_neg)}")
    df_pos= df_pos[df_pos[column] >= split_date]
    df_neg = df_neg[df_neg[column] >= split_date]
    print(f"new sample count: {len(df_pos)+len(df_neg)}")


    df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
    df_neg = df_neg.sample(frac=1)
    df_pos = df_pos.sample(frac=1)

    df_neg = df_neg.reset_index(drop=True)
    df_pos = df_pos.reset_index(drop=True)


    X_neg = []
    y_neg = []
    species_neg = []
    for index, series in df_neg.iterrows():
        y_neg.append(series.y)
        X_neg.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
        if series["Common name"] in species_labels:
            species_neg.append(species_labels[series["Common name"]])
        else:
            species_neg.append(species_labels["other"])
    y_neg = np.array(y_neg)
    X_neg = np.array(X_neg)
    species_neg = np.array(species_neg)
    print(species_neg.shape)


    sequences = df_neg.TruncatedUniProtSequence.tolist()
    sequences = [" ".join(seq) for seq in sequences]
    sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, max_length=33, truncation=True)
    input_ids_neg = np.array(ids['input_ids'])
    

    X_pos = []
    y_pos = []
    species_pos = []
    for index, series in df_pos.iterrows():
        y_pos.append(series.y)
        X_pos.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
        if series["Common name"] in species_labels:
            species_pos.append(species_labels[series["Common name"]])
        else:
            species_pos.append(species_labels["other"])
    y_pos = np.array(y_pos)
    X_pos = np.array(X_pos)
    species_pos = np.array(species_pos)
    print(species_pos.shape)

    sequences = df_pos.TruncatedUniProtSequence.tolist()
    sequences = [" ".join(seq) for seq in sequences]
    sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, max_length=33, truncation=True)
    input_ids_pos = np.array(ids['input_ids'])

    filename = file.split("_")[0]
    file_dir = f"{output_dir}{filename}/"

    os.makedirs(f"{file_dir}indices", exist_ok=True)
    os.makedirs(f"{file_dir}onehot", exist_ok=True)
    os.makedirs(f"{file_dir}input_ids", exist_ok=True)

    np.save(f"{file_dir}indices/X_train_neg.npy", X_neg)
    np.save(f"{file_dir}indices/y_train_neg.npy", y_neg)
    np.save(f"{file_dir}indices/X_train_pos.npy", X_pos)
    np.save(f"{file_dir}indices/y_train_pos.npy", y_pos)

    np.save(f"{file_dir}onehot/X_train_neg.npy", indexListToOneHot(X_neg))
    np.save(f"{file_dir}onehot/y_train_neg.npy", y_neg)
    np.save(f"{file_dir}onehot/X_train_pos.npy", indexListToOneHot(X_pos))
    np.save(f"{file_dir}onehot/y_train_pos.npy", y_pos)

    print(input_ids_neg.shape, input_ids_pos.shape) 
    np.save(f"{file_dir}input_ids/X_train_neg.npy", input_ids_neg)
    np.save(f"{file_dir}input_ids/y_train_neg.npy", y_neg)
    np.save(f"{file_dir}input_ids/X_train_pos.npy", input_ids_pos)
    np.save(f"{file_dir}input_ids/y_train_pos.npy", y_pos)

    np.save(f"{file_dir}species_pos.npy", species_pos)
    np.save(f"{file_dir}species_neg.npy", species_neg)
# %%
