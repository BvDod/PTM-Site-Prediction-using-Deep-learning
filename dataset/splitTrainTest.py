import os
import pandas as pd
import requests as r
import urllib
import io



def main():
    pd.set_option('display.max_rows', 10)

    source_dir = "dataset/data/processed/yearsAdded/"
    output_dir = "dataset/data/processed/final_split/"


    # Get all full UniProt sequences for all dbPTM source files
    for file in os.listdir(source_dir):

        # Check if already processed, else skip
        if file.split("_")[-3] == "neg":
            continue

        file_neg = file.replace('pos', 'neg')

        df_pos = pd.read_csv(f"{source_dir}/{file}", delimiter=",")
        df_neg = pd.read_csv(f"{source_dir}/{file_neg}", delimiter=",")
        
        # column = "DateCreated"
        column = "DateSeqModified"

        df_pos[column] = pd.to_datetime(df_pos[column])
        df_neg[column] = pd.to_datetime(df_neg[column])

        df_pos = df_pos.sort_values(column)

        split_i = int(len(df_pos) * 0.8)
        split_date = df_pos.iloc[split_i][column]

        print(f"{file}: {split_date}")

        df_pos_train = df_pos[df_pos[column] <= split_date]
        df_pos_test = df_pos[df_pos[column] > split_date]
        df_neg_train = df_neg[df_neg[column] <= split_date]
        df_neg_test = df_neg[df_neg[column] > split_date]

        print(f"Train: {len(df_pos_train)}/{len(df_neg_train)}")
        print(f"Test: {len(df_pos_test)}/{len(df_neg_test)}")

        df_pos_train.to_csv(f"{output_dir}/train/{file}_train", index=False)
        df_pos_test.to_csv(f"{output_dir}/test/{file}_test", index=False)
        df_neg_train.to_csv(f"{output_dir}/train/{file_neg}_train", index=False)
        df_neg_test.to_csv(f"{output_dir}/test/{file_neg}_test", index=False)

        print()




    
if __name__ == "__main__":
    main()