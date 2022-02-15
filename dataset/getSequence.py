import os
import pandas as pd
import uniprot
import requests as r
import urllib
import io



def main():
    pd.set_option('display.max_rows', 10)

    dir = "dataset/data/dbPTM-source"
    column_names = ["UniprotID", "UniprotAC", "PTM_location", "PTM_type", "number??", "dbPTMSequence"]

    for file in os.listdir(dir):
        df = pd.read_csv(f"{dir}/{file}", delimiter="\t", names=column_names)

        batch_size = 10000
        df["UniProtSequence"] = None
        for i in range((len(df)//batch_size) + 1):
            i_lower = i*batch_size
            i_upper = ((i+1)*batch_size-1) if not ((i+1)*batch_size > len(df)) else len(df)
            df.loc[i_lower:i_upper, "UniProtSequence"] = getSequenceBatch(df.loc[i_lower:i_upper, "UniprotAC"].tolist())
        
        print(f"Missing sequences: {df['UniProtSequence'].isna().sum()}")
            
        df.to_csv(f"dataset/data/processed/full_sequence/{file}_uniprot_sequence")
        print(f"Removing {df['UniProtSequence'].isna().sum()} PTM's because of missing Uniprot sequences")
        print(f"Removing {df['dbPTMSequence'].isna().sum()} PTM's because of missing dbPTM sequences")
        df = df[df['UniProtSequence'].notna()]
        df = df[df['dbPTMSequence'].notna()]

        df["TruncatedUniProtSequence"] = df.apply(TruncateSequence, axis=1)



def calculateRange(index, seq_length, n=10):
    i_lower = index - (n+1)
    i_upper = index + n
    if index-(n+1) < 0:
        i_lower = 0
    if index+10 > seq_length:
        i_upper = seq_length + 1
    return i_lower, i_upper

def getSequenceBatch(list_of_ACs):
    url = 'https://www.uniprot.org/uploadlists/'  # This is the webserver to retrieve the Uniprot data
    params = {
        'from': "ACC",
        'to': 'ACC',
        'format': 'tab',
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
    df_final = df_final.merge(df_fasta, how="left")
    df_final = df_final.drop(columns=["Query", "Entry"])

    return df_final["UniProtSequence"].tolist()
    

def TruncateSequence(df):
    """
    currentUrl= "http://www.uniprot.org/uniprot/" + df.UniprotAC + ".fasta"
    response = r.post(currentUrl)
    sequence = "".join(response.text.split("\n")[1:])
    """
    sequence = df.UniProtSequence
    
    i = df.PTM_location
    if not type(sequence) == str:
        print(sequence)
    if type(df.dbPTMSequence) == float:
        print(df.dbPTMSequence)
        print(df)
    i_lower, i_upper = calculateRange(i, len(sequence))
    

    if not sequence[i_lower:i_upper] ==  df.dbPTMSequence.replace("-", ""):
        i = sequence.find(df.dbPTMSequence) + 11
        i_lower, i_upper = calculateRange(i, len(sequence))
        
        if not sequence[i_lower:i_upper] ==  df.dbPTMSequence.replace("-", ""):
            print(i_lower, i_upper)
            print(sequence[i_lower:i_upper])
            print(df.UniprotAC, df.UniprotID)
            print(sequence[i_lower:i_upper])
            print(df.dbPTMSequence)

            print(f"        Discarded: {df.UniprotID} ({df.UniprotAC})")
            return None

        print(f"        Fixed by finding substring: {df.UniprotID} ({df.UniprotAC})")
        return sequence
    
    # print(f"Added sequence sucesfully: {df.UniprotID} ({df.UniprotAC})")


def getSequence():
    url = 'https://www.uniprot.org/uploadlists/'  # This is the webserver to retrieve the Uniprot data
    params = {
        'from': "ACC",
        'to': 'ACC',
        'format': 'tab',
        'query': " ".join(uniprot_ids),
        'columns': 'id,sequence'}

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
    df_fasta = pd.read_csv(StringIO(res.decode("utf-8")), sep="\t")
    df_fasta.columns = ["Entry", "Sequence", "Query"]
    # it might happen that 2 different ids for a single query id are returned, split these rows
    return df_fasta.assign(Query=df_fasta['Query'].str.split(',')).explode('Query')



if __name__ == "__main__":
    main()

    12345678910