import os
import pandas as pd
import requests as r
import urllib
import io

def main():

    # For all retrieved full UniProt sequences, now truncate them
    trunc_dir = "dataset/data/processed/truncated"
    full_seq_dir = "dataset/data/processed/full_sequence"

    column_names = ["UniprotID", "UniprotAC", "PTM_location", "PTM_type", "number??", "dbPTMSequence"]
    column_names.append("UniProtSequence")

    for file in os.listdir(full_seq_dir):

        #check if already processed, else skip
        if os.path.exists(f"{trunc_dir}/{file}_truncated"):
            continue

        print(f"Truncatting and checking sequences for file: {file}...")
        df = pd.read_csv(f"{full_seq_dir}/{file}", delimiter=",")
        df = df.apply(TruncateSequence, axis=1, result_type='expand')

        df = df.drop(columns = ["UniProtSequence"])
        df.to_csv(f"{trunc_dir}/{file}_truncated", index=False)
        
        print(df["truncateStatus"].value_counts())


def TruncateSequence(df):
    """ This functions truncates the Uniprot protein to the proper size, also checks if dbPTM and Uniprots sequence are the sam3"""

    sequence = df.UniProtSequence
    i_lower, i_upper, _, _ = calculateRange(df.PTM_location, len(sequence))  # get truncation range
    
    # Check if Uniprot and dbPTM sequence are fully the same
    if not sequence[i_lower:i_upper] ==  df.dbPTMSequence.replace("-", ""):

        # Attempt to re-allign proteins.
        i = sequence.find(df.dbPTMSequence) + 11
        i_lower, i_upper, _, _ = calculateRange(i, len(sequence))        
        if not sequence[i_lower:i_upper] ==  df.dbPTMSequence.replace("-", ""):
            # print(f"        Discarded: {df.UniprotID} ({df.UniprotAC})")
            df["truncateStatus"] = 2
            df["TruncatedUniProtSequence"] = None
            return df

        # Re-allignemnt succesfull
        # print(f"        Fixed by finding substring: {df.UniprotID} ({df.UniprotAC})")
        df["truncateStatus"] = 1
    else:
        df["truncateStatus"] = 0

    df["TruncatedUniProtSequence"] = createTruncatedSequence(sequence, df.PTM_location, n=15)
    return df

def createTruncatedSequence(UniProtSequence, PTM_location, n=10):
    """Create a properly truncated sequence"""

    i_lower, i_upper, discarded_lower, discarded_upper = calculateRange(PTM_location, len(UniProtSequence), n=n)
    truncated_seq = UniProtSequence[i_lower:i_upper]
    pre_dashes = discarded_lower * "-"
    post_dashes = discarded_upper * "-"
    truncatedSequence = "".join([pre_dashes, truncated_seq, post_dashes])
    return truncated_seq


def calculateRange(index, seq_length, n=10):
    "Function used to calcuate lower and upper index range if you want n proteins around the middle AA"

    i_lower, i_upper = index-(n+1), index + n
    discarded_lower, discarded_upper = 0, 0
    if index-(n+1) < 0:
        discarded_lower = abs(index-(n+1))
        i_lower = 0
    if index+n > seq_length:
        discarded_upper = index+n - seq_length
        i_upper = seq_length + 1
    return i_lower, i_upper, discarded_lower, discarded_upper


if __name__ == "__main__":
    main()