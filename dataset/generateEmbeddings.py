# %%

import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
from tqdm import tqdm

import os
import pandas as pd
from math import ceil


def main():

    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained("models/prot_t5_xl_uniref50").half()
    model = model.to(device)
    model = model.eval()
    gc.collect()

    

    # For all retrieved full UniProt sequences, now truncate them
    input_dir = "dataset/data/processed/final_split/train"
    output_dir = "dataset/data/learningData/balanced/train/"

    balance_dataset = ["Ubiquitination", "Phosphorylation-Y", "Acetylation", "Phosphorylation-Y"]

    for file in os.listdir(input_dir):
        if not file.split("_")[-4] == "pos":
            continue

        # check if already processed, else skip
        if os.path.exists(f"{output_dir}/{file}_oneHot"):
            continue
        print(file)

        balance_neg_samples =False
        for string in balance_dataset:
            if string in file:
                balance_neg_samples = True
        
        if balance_neg_samples:
            print("Balancing negative samples")

             
        df_pos = pd.read_csv(f"{input_dir}/{file}", delimiter=",")
        file_neg = file.replace('pos', 'neg')
        df_neg = pd.read_csv(f"{input_dir}/{file_neg}", delimiter=",")
        
        df_pos["y"] = 1
        df_neg["y"] = 0

        n = len(df_pos)
        df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
        
        if balance_neg_samples:
            df_neg = df_neg.sample(n=n)
        else:
            df_neg = df_neg.sample(frac=1)
        df_pos = df_pos.sample(frac=1)

        df_neg = df_neg.reset_index(drop=True)
        df_pos = df_pos.reset_index(drop=True)

        batch_size = 512


        sequences = df_neg.TruncatedUniProtSequence.tolist()
        sequences = [" ".join(seq) for seq in sequences]
        sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        embeddings_list = []
        for batch in range(ceil(len(sequences)/batch_size)):
            print(f"{batch}/{ceil(len(sequences)/batch_size)}")
            if (batch+1)*batch_size > len(sequences):
                sequences_batch = sequences[batch*batch_size:]
            else:
                sequences_batch = sequences[batch*batch_size:(batch + 1)*batch_size] 

            ids = tokenizer.batch_encode_plus(sequences_batch, add_special_tokens=True, padding=True, max_length=33, truncation=True)
            input_ids = torch.tensor(np.array(ids['input_ids']), device=device)
            with torch.no_grad():
                attention_mask=torch.ones_like(input_ids)
                embeddings = model(input_ids=input_ids,attention_mask=attention_mask)
            embeddings = embeddings.last_hidden_state.cpu().numpy()

            embeddings_list.append(embeddings)
        embeddings_neg = np.concatenate(embeddings_list, axis=0)
        y_neg = np.array(df_neg.y.to_list())
 

        sequences = df_pos.TruncatedUniProtSequence.tolist()
        sequences = [" ".join(seq) for seq in sequences]
        sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        embeddings_list = []
        for batch in range(ceil(len(sequences)/batch_size)):
            print(f"{batch}/{ceil(len(sequences)/batch_size)}")
            if (batch+1)*batch_size > len(sequences):
                sequences_batch = sequences[batch*batch_size:]
            else:
                sequences_batch = sequences[batch*batch_size:(batch + 1)*batch_size] 

            ids = tokenizer.batch_encode_plus(sequences_batch, add_special_tokens=True, padding=True, max_length=33, truncation=True)
            input_ids = torch.tensor(np.array(ids['input_ids']), device=device)
            with torch.no_grad():
                attention_mask=torch.ones_like(input_ids)
                embeddings = model(input_ids=input_ids,attention_mask=attention_mask)
            embeddings = embeddings.last_hidden_state.cpu().numpy()
            embeddings_list.append(embeddings)
        embeddings_pos = np.concatenate(embeddings_list, axis=0)
        y_pos = np.array(df_pos.y.to_list())
        print(len(y_pos))
        print(embeddings.shape)



        filename = file.split("_")[0]
        file_dir = f"{output_dir}{filename}/"

 
        os.makedirs(f"{file_dir}embeddings", exist_ok=True)

        np.save(f"{file_dir}embeddings/X_train_neg.npy", embeddings_neg)
        np.save(f"{file_dir}embeddings/y_train_neg.npy", y_neg)
        np.save(f"{file_dir}embeddings/X_train_pos.npy", embeddings_pos)
        np.save(f"{file_dir}embeddings/y_train_pos.npy", y_pos)


if __name__ == "__main__":
    main()