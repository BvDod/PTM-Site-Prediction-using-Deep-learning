import os
import pandas as pd
import requests as r
import urllib
import io



def main():
    pd.set_option('display.max_rows', 10)

    full_seq_dir = "dataset/data/processed/full_sequence"
    dest_dir = "dataset/data/processed/split_AA"

    PTM_AA_to_split = {
        "Hydroxylation": ["K", "P"],
        "S-palmitoylation": ["C", ],
        "Methylation": ["K", "R"],
        "Phosphorylation": [["S", "T"], "Y"]
    }

    for PTMType, AAs in PTM_AA_to_split.items():
        if not os.path.exists(f"{full_seq_dir}/{PTMType}_uniprot_sequence"):
            print(f"Could not find file for {PTMType}")
            continue

        df = pd.read_csv(f"{full_seq_dir}/{PTMType}_uniprot_sequence")
        print(f"Splitting {PTMType} file in seperate files for:{AAs}")
        print(f"Full file contains {len(df)} entries")
        
        for AA in AAs:
            if len(AA) > 1:  
                df_AA = df[df["ModifiedAA"].isin(AA)]
            else:
                df_AA = df[df["ModifiedAA"] == AA]

            df_AA.to_csv(f"{dest_dir}/{PTMType}-{AA}_uniprot_sequence", index=False)
            print(f"Filtered {AA}: {len(df_AA)} samples")
        print()

if __name__ == "__main__":
    main()