# %%
import os
import pandas as pd

# input_dir = "data/processed/final_split/test"
input_dir = "data/processed/final_split_noFiltering/test"
output_dir = "data/Musite_Fasta_noFiltering"

# %%

for file in os.listdir(input_dir):
    # if not file.split("_")[-4] == "pos":
    if not file.split("_")[-1] == "pos":
        continue

    df_pos = pd.read_csv(f"{input_dir}/{file}", delimiter=",")

    file_neg = file.replace('pos', 'neg')
    df_neg = pd.read_csv(f"{input_dir}/{file_neg}", delimiter=",")
    df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
    
    df_pos["y"] = 1
    df_neg["y"] = 0

    df = pd.concat([df_pos, df_neg])
    n = 20000
    if n > len(df):
        n = len(df)
    df = df.sample(n=n)
    filename = file.split("_")[0]

    os.makedirs(f"{output_dir}{filename}/", exist_ok=True)
    df.to_csv(f"{output_dir}{filename}/df_test")

    with open(f"{output_dir}{filename}/PTM.fasta", "w+") as f:
        with open(f"{output_dir}{filename}/labels.txt", "w+") as f_label:
            for i, (index, row) in enumerate(df.iterrows()):
                f.write(f">{i}\n")
                f.write(f"{row['TruncatedUniProtSequence']}\n")
                f_label.write(f"{row['y']}\n")