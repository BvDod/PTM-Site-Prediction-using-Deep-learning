import os
import pandas as pd
from Bio import SeqIO


def main():
    source_dir = "dataset/data/processed/truncated/"
    source_dir2 = "dataset/data/processed/negative_samples/"

    output_dir = "dataset/data/processed/fasta_post_70/"
    output_non_redundant = "dataset/data/processed/non_redundant_70/"

    
    for file in os.listdir(output_dir):

            print(f"Processing post-fasta file for: {file}...")
            non_fasta_file = file.rsplit(sep="_", maxsplit=1)[0]

            # Positve samples
            df_pos = pd.read_csv(f"{source_dir}/{non_fasta_file}", delimiter=",")
            df_pos = df_pos[df_pos["truncateStatus"] != 2]

            # Negative samples
            df_neg = pd.read_csv(f"{source_dir2}/{non_fasta_file}_negative", delimiter=",")

            filename = f"{output_dir}{file}"
            pos_indexes, neg_indexes = read_fasta_file(filename)
            
            df_pos_post = df_pos[df_pos.index.isin(pos_indexes)]
            df_neg_post = df_neg[df_neg.index.isin(neg_indexes)]

            print(f"Positive samples: {len(df_pos)} > {len(df_pos_post)}")
            print(f"Negative samples: {len(df_neg)} > {len(df_neg_post)}")

            df_pos_post.to_csv(f"{output_non_redundant}{non_fasta_file}_nonRed_pos", index=False)
            df_neg_post.to_csv(f"{output_non_redundant}{non_fasta_file}_nonRed_neg", index=False)

            print()

        


def read_fasta_file(filename):
    positive_samples, negative_samples = [], []
    fasta_sequences = SeqIO.parse(open(filename),'fasta')
    for data in fasta_sequences:
        pos_or_neg, index = data.name.split("_")
        if pos_or_neg == "positive":
            positive_samples.append(int(index))
        elif pos_or_neg == "negative":
            negative_samples.append(int(index))
        else:
            print("Parse error: not neg or pos")
            exit()
    return positive_samples, negative_samples
    

        



if __name__ == '__main__':
    main()
