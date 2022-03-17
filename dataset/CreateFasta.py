import os
import pandas as pd


def main():
    source_dir = "dataset/data/processed/truncated/"
    source_dir2 = "dataset/data/processed/negative_samples/"

    output_dir = "dataset/data/processed/fasta_pre/"

    
    for file in os.listdir(source_dir):

            # Check if already processed, else skip
            if os.path.exists(f"dataset/data/processed/fasta_pre/{file}_pre.fasta"):
                continue
                
            print(f"Creating fasta file for: {file}...")

            # Positve samples
            df = pd.read_csv(f"{source_dir}/{file}", delimiter=",")
            df = df[df["truncateStatus"] == 0]

            names, sequences = df.index.tolist(), df["TruncatedUniProtSequence"].tolist()
            names = [f"positive_{name}" for name in names]

            # Negative samples
            df = pd.read_csv(f"{source_dir2}/{file}_negative", delimiter=",")
            
            print(len(df))
            names2, sequences2 = df.index.tolist(), df["dbPTMSequence"].tolist()
            names2 = [f"negative_{name}" for name in names2]



            filename = f"{output_dir}{file}_pre.fasta"
            create_fasta_file(filename, names+names2, sequences+sequences2)
        


def create_fasta_file(filename, names, sequences):
    with open(filename, "w") as fasta_file:
        assert(len(names) == len(sequences))
        for i in range(len(names)):
            fasta_file.write(">" + names[i] + "\n" + sequences[i].replace("-", "") + "\n")



if __name__ == '__main__':
    main()
