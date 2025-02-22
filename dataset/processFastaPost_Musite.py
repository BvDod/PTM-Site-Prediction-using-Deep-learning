import os
import pandas as pd
from Bio import SeqIO


def main():

    source_pos = "dataset/data/processed/final_split/Musite/Musite_ST_pos"
    source_neg = "dataset/data/processed/final_split/Musite/Musite_ST_neg"

    output = "dataset/data/processed/fasta_post_musite/ST_pre.fasta.clstr"

    # Positve samples
    df_pos = pd.read_csv(source_pos, delimiter=",")

    # Negative samples
    df_neg = pd.read_csv(source_neg, delimiter=",")

    pos_indexes, neg_indexes = read_fasta_cluster_file(output)
    
    df_pos_post = df_pos[df_pos.index.isin(pos_indexes)]
    df_neg_post = df_neg[df_neg.index.isin(neg_indexes)]

    print(f"Positive samples: {len(df_pos)} > {len(df_pos_post)}")
    print(f"Negative samples: {len(df_neg)} > {len(df_neg_post)}")

    filename = output.split("/")[-1].split(".")[0]
    df_pos_post.to_csv(f"dataset/data/processed/non_redundant_Musite/{filename}_pos.csv", index=False)
    df_neg_post.to_csv(f"dataset/data/processed/non_redundant_Musite/{filename}_neg.csv", index=False)



        


def read_fasta_file(filename):
    positive_test_samples, negative_test_samples = [], []
    dontinclude_pos, dontinclude_neg = [], []

    fasta_sequences = SeqIO.parse(open(filename),'fasta')
    for data in fasta_sequences:
        pos_or_neg, train_or_test, index = data.name.split("_")
        if (pos_or_neg == "positive") and (train_or_test == "test"):
            positive_test_samples.append(int(index))
        elif (pos_or_neg == "negative") and (train_or_test == "test"):
            negative_test_samples.append(int(index))
    return positive_test_samples, negative_test_samples


def read_fasta_cluster_file(filename):
    positive_test_samples, negative_test_samples = [], []
    dontinclude_pos, dontinclude_neg = set(), set()

    with open(filename, "r") as file_annotated:
        lines = file_annotated.readlines()

        sample_to_add = None
        skip_to_next_cluster = False
        for line in lines:
            if line[0] == ">":

                if sample_to_add:
                    if "negative" in sample_to_add:
                        index = int(sample_to_add.split("_")[-1])
                        if not index in dontinclude_neg:
                            negative_test_samples.append(index)
                    if "positive" in sample_to_add:
                        if not index in dontinclude_pos:
                            positive_test_samples.append(index)
                    sample_to_add = None
                    skip_to_next_cluster = False

            elif skip_to_next_cluster:
                continue

            else:
                name = line.split(">")[1].split(".")[0]
                if line[0] == "0":
                    if "test" in name:
                        sample_to_add = name
                    if "train" in name:
                        skip_to_next_cluster = True
                else:
                    if "train" in name:
                        skip_to_next_cluster = True
                        index = int(sample_to_add.split("_")[-1])
                        if "negative" in name:
                            dontinclude_neg.add(index)
                        if "positive" in name:
                            dontinclude_pos.add(index)

                

    
    
        pos_or_neg, train_or_test, index = line.split("_")
        if (pos_or_neg == "positive") and (train_or_test == "test"):
            positive_test_samples.append(int(index))
        elif (pos_or_neg == "negative") and (train_or_test == "test"):
            negative_test_samples.append(int(index))
    return positive_test_samples, negative_test_samples
    

        



if __name__ == '__main__':
    main()
