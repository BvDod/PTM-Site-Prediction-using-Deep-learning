import os
import pandas as pd
import requests as r
import urllib
import io



def main():
    pd.set_option('display.max_rows', 10)

    full_seq_dir = "dataset/data/processed/yearsAddedNoFiltering/"
    batch_size = 50000

    file_pos = "Phosphorylation-Y_uniprot_sequence_truncated"
    file_neg = "Phosphorylation-Y_uniprot_sequence_truncated_negative"
    df_pos = pd.read_csv(f"dataset/data/processed/truncated/{file_pos}", delimiter=",")
    df_neg = pd.read_csv(f"dataset/data/processed/negative_samples/{file_neg}", delimiter=",")

    print(len(df_pos), len(df_neg))
    df_neg = df_neg.sample(n=int(20000*(len(df_neg)/len(df_pos))))
    df_neg["y"] = 0

    df_pos = df_pos.sample(n=20000)
    df_pos["y"] = 1
    
    print(len(df_pos), len(df_neg))

    df = pd.concat([df_pos, df_neg])
    df.reset_index(drop=True, inplace=True)


    
    # Get all UniProt sequences by retrieving them in batches
    df["DateCreated"] = None
    df["DateSeqModified"] = None

    for i in range((len(df)//batch_size) + 1):
        print(f"{i*batch_size}/{len(df)}")
        i_lower = i*batch_size
        i_upper = ((i+1)*batch_size-1) if not ((i+1)*batch_size > len(df)) else len(df)
        print(i_lower, i_upper)
        df.loc[i_lower:i_upper, "DateCreated"], df.loc[i_lower:i_upper, "DateSeqModified"] = getSequenceBatch(df.loc[i_lower:i_upper, "UniprotAC"].tolist())
    
    print(f"Removing {df['DateCreated'].isna().sum()} PTM's because of missing Uniprot sequences")
    
    # Save
    df.to_csv(f"{full_seq_dir}/{file_pos}_year_added", index=False)
    print(f"Retrieved {len(df)} sequences from UniProt")
    print()


def getSequenceBatch(list_of_ACs):
    """ This function retrieves a batch of sequences from a list of Uniprot AC's"""

    url = 'https://www.uniprot.org/uploadlists/'  # This is the webserver to retrieve the Uniprot data
    params = { 'from': "ACC", 'to': 'ACC', 'format': 'tab',
        'query': " ".join(list_of_ACs),
        'columns': 'id,created,sequence-modified'}

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
    df_fasta = pd.read_csv(io.StringIO(res.decode("utf-8")), sep="\t")
    df_fasta.columns = ["Entry", "DateCreated", "DateSeqModified", "Query"]

    # it might happen that 2 different ids for a single query id are returned, split these rows
    df_fasta = df_fasta.assign(Query=df_fasta['Query'].str.split(',')).explode('Query')
    
    df_final = pd.DataFrame(list_of_ACs, columns=["Query"])

    df_fasta = df_fasta.drop_duplicates(subset="Query", keep="first")
    df_final = df_final.merge(df_fasta, how="left")
    df_final = df_final.drop(columns=["Query", "Entry"])


    return df_final["DateCreated"].tolist(), df_final["DateSeqModified"].tolist()
    

    
if __name__ == "__main__":
    main()