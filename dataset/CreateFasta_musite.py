import os
import pandas as pd


def main():
    source1 = "dataset/data/processed/final_split/Musite/Musite_Y_pos"
    source2 = "dataset/data/processed/final_split/Musite/Musite_Y_neg"
    source3 = "dataset/data/processed/final_split_if_2010/train/Phosphorylation-Y_uniprot_sequence_truncated_nonRed_pos_year_added_train"
    source4 = "dataset/data/processed/final_split_if_2010/train/Phosphorylation-Y_uniprot_sequence_truncated_nonRed_neg_year_added_train"
    source5 = "dataset/data/processed/final_split/Musite_train/Musite_Y_pos.csv"
    source6 = "dataset/data/processed/final_split/Musite_train/Musite_Y_neg.csv"
    
    output = "dataset/data/processed/fasta_pre/Y_pre_incMusite.fasta"

    # Pos samples test
    df = pd.read_csv(source1, delimiter=",")

    names, sequences = df.index.tolist(), df["dbPTMSequence"].tolist()
    names = [f"positive_test_{name}" for name in names]

    # Neg samples test
    df = pd.read_csv(source2, delimiter=",")
    
    print(len(df))
    names2, sequences2 = df.index.tolist(), df["dbPTMSequence"].tolist()
    names2 = [f"negative_test_{name}" for name in names2]

    # pos samples train
    df = pd.read_csv(source3, delimiter=",")

    names3, sequences3 = df.index.tolist(), df["TruncatedUniProtSequence"].tolist()
    names3 = [f"positive_train_{name}" for name in names3]

    # neg samples train
    df = pd.read_csv(source4, delimiter=",")

    names4, sequences4 = df.index.tolist(), df["dbPTMSequence"].tolist()
    names4 = [f"negative_train_{name}" for name in names4]


    # pos samples mus train
    df = pd.read_csv(source5, delimiter=",")
    
    print(len(df))
    names5, sequences5 = df.index.tolist(), df["dbPTMSequence"].tolist()
    names5 = [f"positive_trainMus_{name}" for name in names5]


    # neg samples mus train
    df = pd.read_csv(source6, delimiter=",")
    
    print(len(df))
    names6, sequences6 = df.index.tolist(), df["dbPTMSequence"].tolist()
    names6 = [f"negative_trainMus_{name}" for name in names6]


    filename = output
    create_fasta_file(filename, names+names2+names3+names4+names5+names6, sequences+sequences2+sequences3+sequences4+sequences5+sequences6)
        


def create_fasta_file(filename, names, sequences):
    with open(filename, "w") as fasta_file:
        print(len(names), len(sequences))
        assert(len(names) == len(sequences))
        for i in range(len(names)):
            fasta_file.write(">" + names[i] + "\n" + sequences[i].replace("-", "") + "\n")



if __name__ == '__main__':
    main()
