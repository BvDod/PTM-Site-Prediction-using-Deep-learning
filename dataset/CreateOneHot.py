import os
import pandas as pd
import requests as r
import urllib
import io
import numpy as np

def main():

    oneHotEncoding = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
                'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 
                'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 
                'Z': 25, "-": 26}
    

    # For all retrieved full UniProt sequences, now truncate them
    input_dir = "dataset/data/processed/non_redundant_50"
    output_dir = "dataset/data/learningData/balanced/oneHot_50/"

    balance_dataset = False

    for file in os.listdir(input_dir):
        print(file)

        if not file.split("_")[-1] == "pos":
            continue

        # check if already processed, else skip
        if os.path.exists(f"{output_dir}/{file}_oneHot"):
            continue

        print(f"Creating OneHot for: {file}...")
        df_pos = pd.read_csv(f"{input_dir}/{file}", delimiter=",")
        non_pos_file = file.rsplit(sep="_", maxsplit=1)[0]
        df_neg = pd.read_csv(f"{input_dir}/{non_pos_file}_neg", delimiter=",")
        
        df_pos["y"] = 1
        df_neg["y"] = 0

        n = len(df_pos)

        if balance_dataset:
            df_neg = df_neg.sample(n=n)

            df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
            df = pd.concat([df_pos, df_neg])
            df = df.sample(frac=1)

            X = []
            y = []

            df = df.reset_index(drop=True)
            for index, series in df.iterrows():
                y.append(series.y)
                X.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
                

            y = np.array(y)
            X = np.array(X)
            
            filename = file.split("_")[0]
            file_dir = f"{output_dir}{filename}/"

            os.makedirs(file_dir, exist_ok=True)
            np.save(f"{file_dir}X_train.npy", X)
            np.save(f"{file_dir}y_train.npy", y)

        else:
            df_neg["TruncatedUniProtSequence"] = df_neg.dbPTMSequence
            df_neg = df_neg.reset_index(drop=True)
            df_pos = df_pos.reset_index(drop=True)


            X_neg = []
            y_neg = []
            for index, series in df_neg.iterrows():
                y_neg.append(series.y)
                X_neg.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
            y_neg = np.array(y_neg)
            X_neg = np.array(X_neg)


            X_pos = []
            y_pos = []
            for index, series in df_pos.iterrows():
                y_pos.append(series.y)
                X_pos.append([oneHotEncoding[char] for char in series.TruncatedUniProtSequence])
            y_pos = np.array(y_pos)
            X_pos = np.array(X_pos)

            filename = file.split("_")[0]
            file_dir = f"{output_dir}{filename}/"

            os.makedirs(file_dir, exist_ok=True)
            np.save(f"{file_dir}X_train_neg.npy", X_neg)
            np.save(f"{file_dir}y_train_neg.npy", y_neg)
            np.save(f"{file_dir}X_train_pos.npy", X_pos)
            np.save(f"{file_dir}y_train_pos.npy", y_pos)


if __name__ == "__main__":
    main()