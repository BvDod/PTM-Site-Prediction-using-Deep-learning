import os
import pandas as pd
import requests as r
import urllib
import io
import numpy as np
from transformers import BertTokenizer
import re



def indexListToOneHot(input_array):
    """This function converts an array of categorical features to an array of one-hot represented features"""

    OneHotVariables = 27
    samples, columns = input_array.shape

    output_array = np.zeros((samples, OneHotVariables*columns))
    for i in range(samples):
        for j in range(columns):
            output_array[i, (j*OneHotVariables) + input_array[i,j]] = 1
    return output_array

def main():

    oneHotEncoding = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
                'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 
                'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 
                'Z': 25, "-": 26}
    
    
    tokenizer = tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    # [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

    

    # For all retrieved full UniProt sequences, now truncate them
    input_dir = "dataset/data/processed/final_split/train"
    output_dir = "dataset/data/learningData/balanced/train/"

    balance_dataset = False

    for file in os.listdir(input_dir):
        if not file.split("_")[-4] == "pos":
            continue

        # check if already processed, else skip
        if os.path.exists(f"{output_dir}/{file}_oneHot"):
            continue

        print(file)
        
        
 
        df_pos = pd.read_csv(f"{input_dir}/{file}", delimiter=",")
        file_neg = file.replace('pos', 'neg')
        df_neg = pd.read_csv(f"{input_dir}/{file_neg}", delimiter=",")
        
        df_pos["y"] = 1
        df_neg["y"] = 0

        n = len(df_pos)

        df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
        df_neg = df_neg.sample(frac=1)
        df_pos = df_pos.sample(frac=1)

        df_neg = df_neg.reset_index(drop=True)
        df_pos = df_pos.reset_index(drop=True)


        X_neg = []
        y_neg = []
        for index, series in df_neg.iterrows():
            y_neg.append(series.y)
            X_neg.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
        y_neg = np.array(y_neg)
        X_neg = np.array(X_neg)


        sequences = df_neg.TruncatedUniProtSequence.tolist()
        sequences = [" ".join(seq) for seq in sequences]
        sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, max_length=33, truncation=True)
        input_ids_neg = np.array(ids['input_ids'])
    
        

        X_pos = []
        y_pos = []
        for index, series in df_pos.iterrows():
            y_pos.append(series.y)
            X_pos.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
        y_pos = np.array(y_pos)
        X_pos = np.array(X_pos)

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


if __name__ == "__main__":
    main()