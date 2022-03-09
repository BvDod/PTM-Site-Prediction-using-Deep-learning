# %%

import torch
from transformers import BertModel, BertTokenizer
import re
import numpy as np
import gc
from tqdm import tqdm

import os
import pandas as pd





# For all retrieved full UniProt sequences, now truncate them
output_dir = "dataset/data/processed/truncated"
input_dir = "dataset/data/processed/full_sequence"
df_unique_proteins = None
for file in os.listdir(input_dir):


    print(f"Truncatting and checking sequences for file: {file}...")
    df = pd.read_csv(f"{input_dir}/{file}", delimiter=",")
    df = df.drop_duplicates(subset='UniprotAC')[['UniprotAC', "UniProtSequence"]]

    if df_unique_proteins is None:
        df_unique_proteins = df
    else:
        df_unique_proteins = pd.concat([df, df_unique_proteins]).drop_duplicates(subset='UniprotAC')

print("Extracted unique proteins from all PTM types")        


tokenizer = tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").half()
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

sequences = df_unique_proteins["UniProtSequence"].tolist()[:1000]
sequences = [" ".join(seq) for seq in sequences]
sequences= [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, max_length=33, truncation=True)

input_ids = torch.tensor(ids['input_ids'])
attention_mask = torch.tensor(ids['attention_mask'])

# %%
print("Creating Embeddings..")

print(input_ids.shape)



# %%
embeddings = []
for i in tqdm(range(len(input_ids))):
    with torch.no_grad():
        print(input_ids[i])
        print(input_ids[i].shape)
        embedding = model(input_ids=input_ids[i:i+500].to(device), attention_mask=attention_mask[i:i+500].to(device))
        embedding = embedding.last_hidden_state.cpu().numpy()
        seq_len = (attention_mask[i] == 1).sum()
        seq_emd = embedding[:seq_len-1]
        print(seq_len)
        embeddings.append(seq_emd)
        
print(embeddings)
