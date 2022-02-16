import os
import pandas as pd

def main():

    # For all retrieved full UniProt sequences, now truncate them
    dir = "dataset/data/processed/truncated"

    column_names = ["UniprotID", "UniprotAC", "PTM_location", "PTM_type", "number??", "dbPTMSequence"]
    column_names.append("UniProtSequence")

    for file in os.listdir(dir):
        df = pd.read_csv(f"{dir}/{file}")
        print(file)
        print(df["ModifiedAA"].value_counts())
        print()

if __name__ == "__main__":
    main()