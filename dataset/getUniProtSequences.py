import os
import pandas as pd
import requests as r
import urllib
import io



def main():
    pd.set_option('display.max_rows', 10)

    source_dir = "dataset/data/dbPTM-source"
    full_seq_dir = "dataset/data/processed/full_sequence"

    column_names = ["UniprotID", "UniprotAC", "PTM_location", "PTM_type", "number??", "dbPTMSequence"]
    batch_size = 50000


    # Get all full UniProt sequences for all dbPTM source files
    for file in os.listdir(source_dir):

        # Check if already processed, else skip
        if os.path.exists(f"{full_seq_dir}/{file}_uniprot_sequence"):
            continue

        print(f"Getting Uniprot sequences for file: {file}...")
        df = pd.read_csv(f"{source_dir}/{file}", delimiter="\t", names=column_names)

        
        # Get all UniProt sequences by retrieving them in batches
        df["UniProtSequence"] = None
        for i in range((len(df)//batch_size) + 1):
            print(f"{i*batch_size}/{len(df)}")
            i_lower = i*batch_size
            i_upper = ((i+1)*batch_size-1) if not ((i+1)*batch_size > len(df)) else len(df)
            df.loc[i_lower:i_upper, "UniProtSequence"] = getSequenceBatch(df.loc[i_lower:i_upper, "UniprotAC"].tolist())
        
        print(f"Removing {df['UniProtSequence'].isna().sum()} PTM's because of missing Uniprot sequences")
        print(f"Removing {df['dbPTMSequence'].isna().sum()} PTM's because of missing dbPTM sequences")
        df = df[df['UniProtSequence'].notna()]
        df = df[df['dbPTMSequence'].notna()]

        # Add modified AA
        df["ModifiedAA"] = df.apply(lambda x: x["dbPTMSequence"][len(x["dbPTMSequence"])//2], axis=1)
        df.to_csv(f"{full_seq_dir}/{file}_uniprot_sequence", index=False)
        print(f"Retrieved {len(df)} sequences from UniProt")
        print()


def getSequenceBatch(list_of_ACs):
    """ This function retrieves a batch of sequences from a list of Uniprot AC's"""

    url = 'https://www.uniprot.org/uploadlists/'  # This is the webserver to retrieve the Uniprot data
    params = { 'from': "ACC", 'to': 'ACC', 'format': 'tab',
        'query': " ".join(list_of_ACs),
        'columns': 'id,sequence'}

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
    df_fasta = pd.read_csv(io.StringIO(res.decode("utf-8")), sep="\t")
    df_fasta.columns = ["Entry", "UniProtSequence", "Query"]
    # it might happen that 2 different ids for a single query id are returned, split these rows
    df_fasta = df_fasta.assign(Query=df_fasta['Query'].str.split(',')).explode('Query')
    
    df_final = pd.DataFrame(list_of_ACs, columns=["Query"])

    df_fasta = df_fasta.drop_duplicates(subset="Query", keep="first")
    df_final = df_final.merge(df_fasta, how="left")
    df_final = df_final.drop(columns=["Query", "Entry"])


    return df_final["UniProtSequence"].tolist()
    

    
if __name__ == "__main__":
    main()