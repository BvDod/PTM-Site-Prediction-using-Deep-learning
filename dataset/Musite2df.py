import os
import pandas as pd



def createTruncatedSequence(UniProtSequence, PTM_location, n=10):
    """Create a properly truncated sequence"""

    i_lower, i_upper, discarded_lower, discarded_upper = calculateRange(PTM_location, len(UniProtSequence), n=n)
    truncated_seq = UniProtSequence[i_lower:i_upper]
    pre_dashes = discarded_lower * "-"
    post_dashes = discarded_upper * "-"
    truncatedSequence = "".join([pre_dashes, truncated_seq, post_dashes])
    return truncatedSequence


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


def getPtmSequences(aa_string, indexes):
    sequences = []
    indexes = [index + 1 for index in indexes]
    for index in indexes:
        sequences.append(createTruncatedSequence(aa_string, index, n=16))
    return sequences

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]




input_file_annotated = "Musitedeep/test_allspecies_sequences_annotated_ST.fasta"
PTMType = "Phosphorylation-ST"

legal_aas = ["S", "T"]


df_dict_pos = {"UniprotID": [], "UniprotAC": [], "dbPTMSequence": []}
df_dict_neg = {"UniprotID": [], "UniprotAC": [], "dbPTMSequence": []}
with open(input_file_annotated, "r") as file_annotated:
    lines_annotated = file_annotated.readlines()

    UniprotID, UniprotAC = None, None
    aa_string = ""
    positive_locations = []
    negative_locations = []
    for linenr, line in enumerate(lines_annotated):
        if line[0] == ">":
            if not linenr == 0:
                sequences_pos = getPtmSequences(aa_string, positive_locations)
                sequences_neg = getPtmSequences(aa_string, negative_locations)
                for sequence in sequences_pos:
                    df_dict_pos["UniprotID"].append(UniprotID)
                    df_dict_pos["UniprotAC"].append(UniprotAC)
                    df_dict_pos["dbPTMSequence"].append(sequence)
                for sequence in sequences_neg:
                    df_dict_neg["UniprotID"].append(UniprotID)
                    df_dict_neg["UniprotAC"].append(UniprotAC)
                    df_dict_neg["dbPTMSequence"].append(sequence)
            UniprotID, UniprotAC = line.split("|")[2].split(" ")[0], line.split("|")[1] 
            positive_locations = []
            negative_locations = []
            aa_string = ""
            continue
        positive_indexes = find(line, "#")
        positive_indexes = [positive_indexes[i] - (i+1) for i in range(len(positive_indexes))]
        line_stripped = line.rstrip().replace('#', '')
        for aa in legal_aas:
            negative_indexes = find(line_stripped, aa)
            negative_indexes = [negative_index + len(aa_string) for negative_index in negative_indexes if negative_index not in positive_indexes]
            negative_locations = negative_locations + negative_indexes
        positive_indexes = [positive_index + len(aa_string) for positive_index in positive_indexes]
        positive_locations = positive_locations + positive_indexes
        aa_string = aa_string + line_stripped
        
    
    sequences_pos = getPtmSequences(aa_string, positive_locations)
    sequences_neg = getPtmSequences(aa_string, negative_locations)
    for sequence in sequences_pos:
        df_dict_pos["UniprotID"].append(UniprotID)
        df_dict_pos["UniprotAC"].append(UniprotAC)
        df_dict_pos["dbPTMSequence"].append(sequence)
    for sequence in sequences_neg:
        df_dict_neg["UniprotID"].append(UniprotID)
        df_dict_neg["UniprotAC"].append(UniprotAC)
        df_dict_neg["dbPTMSequence"].append(sequence)

df_pos = pd.DataFrame.from_dict(df_dict_pos)
df_pos.to_csv("test_output_pos.csv", index=False)

df_neg = pd.DataFrame.from_dict(df_dict_neg)
df_neg.to_csv("test_output_neg.csv", index=False)







