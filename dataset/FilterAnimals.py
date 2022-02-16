import os
import pandas as pd
import requests as r
import urllib
import io

def main():
    pd.set_option('display.max_rows', 10)

    source_dir = "dataset/data/processed/truncated"
    output_dir = "dataset/data/processed/animals"

    column_names = ["UniprotID", "UniprotAC", "PTM_location", "PTM_type", "number??", "dbPTMSequence"]
    batch_size = 50000


    # Get all full UniProt sequences for all dbPTM source files
    for file in os.listdir(source_dir):

        # Check if already processed, else skip
        if os.path.exists(f"{output_dir}/{file}_animals"):
            continue

        print(f"Filtering animals for file: {file}...")
        df = pd.read_csv(f"{source_dir}/{file}", delimiter=",")
        # df_uniprot_animals = pd.read_excel("dataset/data/uniprot_animals.xlsx")

        df_uniprot_animals = pd.read_csv("dataset/data/uniprot_animals.csv")

        print(f"Total samples: {len(df)}")
        df = df[df['UniprotID'].str.split('_').str[-1].isin(df_uniprot_animals["Mnemonic"])]
        print(f"Animal samples: {len(df)}")
        df.to_csv(f"{output_dir}/{file}_animals")
        print()


if __name__ == "__main__":
    main()