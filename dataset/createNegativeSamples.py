import os
import pandas as pd
import requests as r
import urllib
import io

from truncateAndCheck import calculateRange, createTruncatedSequence



def main():
    pd.set_option('display.max_rows', 10)

    source_dir = "dataset/data/processed/full_sequence"
    dest_dir = "dataset/data/processed/negative_samples"
    cleaned_dir = "dataset/data/processed/truncated"
    negative_dir = "dataset/data/processed/negative_samples"

    for file in os.listdir(source_dir):

        # Check if already processed, else skip
        if os.path.exists(f"{dest_dir}/{file}_negative_samples"):
            continue

        print(f"Creating negative samples for file: {file}...")
        df = pd.read_csv(f"{source_dir}/{file}", delimiter=",")

        # Remove invalid sequences
        df_truncated_cleaned = pd.read_csv(f"{cleaned_dir}/{file}_truncated", delimiter=",")
        df = df[df.UniprotAC.isin(df_truncated_cleaned[df_truncated_cleaned["truncateStatus"]==1]["UniprotAC"]) | df.UniprotAC.isin(df_truncated_cleaned[df_truncated_cleaned["truncateStatus"]==0]["UniprotAC"])]


        unique_df = df.drop_duplicates(subset=["UniProtSequence", "ModifiedAA"])
        
        print(f"Positive samples: {len(df)}")

        negative_samples_df = pd.DataFrame()

        i = 0
        negativeSamplesDicts = []
        for index, series in unique_df.iterrows():
            AA = series.ModifiedAA
            AC = series.UniprotAC
            uniProtSeq = series.UniProtSequence

            negative_samples = find(uniProtSeq, AA)
            positive_samples = df[(df["UniprotAC"] == AC) & (df["ModifiedAA"] == AA)]["PTM_location"].tolist()
            negative_samples = [negative_sample for negative_sample in negative_samples if negative_sample not in positive_samples]

            for PTM_location in negative_samples:
                i += 1
                truncatedSequence = createTruncatedSequence(uniProtSeq, PTM_location, n=15)
                negativeSamplesDicts.append({'UniprotID': series.UniprotID, 'UniprotAC': AC, 'PTM_location': PTM_location,
                                             'PTM_type':series.PTM_type, 'number??': series["number??"],
                                             'dbPTMSequence': truncatedSequence, 'ModifiedAA': AA})
                
                if i > 50000:
                    i = 0
                    negative_samples_df = negative_samples_df.append(negativeSamplesDicts)
                    negativeSamplesDicts = []
        negative_samples_df = negative_samples_df.append(negativeSamplesDicts)
        print(f"Negative samples: {len(negative_samples_df)}")
        print()
        negative_samples_df.to_csv(f"{negative_dir}/{file}_truncated_negative", index=False)
            

def find(s, ch):
    return [i + 1 for i, ltr in enumerate(s) if ltr == ch]

if __name__ == "__main__":
    main()