from multiprocessing.connection import wait
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy

from math import sqrt


df =  pd.read_csv("embedding.csv")
df["N"] = 25
df["CI"] = 0

sns.set_theme()
sns.set_style("ticks")
sns.set_palette("deep")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from decimal import Decimal


save = True

def grouped_barplot(df, cat, subcat, val , err, title, range_graph, figsize, tilted=True, legend_outside=False, add_p = False):
    
    if add_p:
        df["pstring"] = ""
        amino_acids = df[cat].unique()
        x = np.arange(len(amino_acids))

        treatments = df[subcat].unique()
        offsets = (np.arange(len(treatments))-np.arange(len(treatments)).mean())/(len(treatments)+1.)

        for i,gr in enumerate(amino_acids):
            print(gr)
            dfg = df[df[cat] == gr]
            index_to_compare = np.argmax(dfg[val].values)

            compare_to = set(range(len(dfg[val].values))) - set([index_to_compare,])

            for i in compare_to:
                p = t_test(dfg[val].values[index_to_compare], dfg["std AUC ROC"].values[index_to_compare], dfg[val].values[i], dfg["std AUC ROC"].values[i])
                print(f"{treatments[index_to_compare]} vs {treatments[i]} = {p}")
                if p < 0.001:
                    string = f"p = {p:.3e}"
                else:
                    string = f"p = {p:.3f}"
                
                indexes = df[df[cat] == gr].index.values

                df.loc[indexes[i], "pstring"] = string
                
                # print(df[df[cat] == gr].iloc[[i,]])


            
            
    
    
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()

    plt.figure(figsize=figsize)
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label=f"{gr}")
        plt.errorbar(x+offsets[i], dfg[val].values-0.00125, yerr=dfg[err].values , fmt = "none", color="black")

    plt.xlabel(cat)
    plt.ylabel(val)

    
    if tilted:
        plt.xticks(x, u, rotation=-40, ha="left")
    else:
        plt.xticks(x, u, )
    plt.yticks()

    
    plt.ylim(*range_graph)

    if legend_outside:
        plt.legend(loc='upper left', bbox_to_anchor=(1.025, 1.0), prop={'size': 9})
    else:
        plt.legend(prop={'size': 9})
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig(f"figures/{title}.png", bbox_inches = "tight", dpi=300)
    else:
        plt.show()


def t_test(mean1, std1, mean2, std2):
    import math
    std = math.sqrt(((std1**2)+(std2**2))/2)
    t = (abs(mean1-mean2))/(math.sqrt(((1/25) + (1/4))*(std**2)))
    p = scipy.stats.t.sf(t, 24)
    # print(std1, std2)
    return p

def t_test_on_df(df, cat, subcat, mean, std):



    print(subcat)
    amino_acids = df[cat].unique()
    x = np.arange(len(amino_acids))

    treatments = df[subcat].unique()
    offsets = (np.arange(len(treatments))-np.arange(len(treatments)).mean())/(len(treatments)+1.)

    
    for i,gr in enumerate(amino_acids):
        print()
        print(gr)
        dfg = df[df[cat] == gr]
        index_to_compare = np.argmax(dfg[mean].values)
        compare_to = set(range(len(dfg[mean].values))) - set([index_to_compare,])
        for i in compare_to:
            p = t_test(dfg[mean].values[index_to_compare], dfg[std].values[index_to_compare], dfg[mean].values[i], dfg[std].values[i])
            print(f"{treatments[index_to_compare]} vs {treatments[i]} = {p}")
    print()
    print()
    print()
    return None





cat = "AA"
subcat = "EmbeddingType"
val = "avg  AUC ROC"
err = "CI"
title = "AA representation - Cross-validation"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5], add_p = True)







df =  pd.read_csv("sampling.csv")
df["CI"] = df["std AUC ROC"]/sqrt(25)
cat = "AA"
subcat = "Sampling Method"
val = "avg  AUC ROC"
err = "CI"
title = "Data imbalance - Cross-validation"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5], add_p = True)







df =  pd.read_csv("arch2.csv")
df["CI"] = df["std AUC ROC"]/sqrt(25)
cat = "AA"
subcat = "Architecture"
val = "avg  AUC ROC"
err = "CI"
title = "Model Architecture - Cross-validation"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5], add_p = True)






df1 = pd.read_csv("protBert.csv")
df1["EmbeddingType"] = "ProtBert emb."
df2 = pd.read_csv("singletests.csv")
df2["EmbeddingType"] = "Adaptive emb."
df = pd.concat([df2, df1])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = df["std AUC ROC"]/sqrt(25)


cat = "AA"
subcat = "EmbeddingType"
val = "avg AUC ROC"
err = "CI"
title = "Pre-trained LM Embeddings - Cross-validation"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5], tilted=True, legend_outside=False, add_p = True)



df1 = pd.read_csv("protBert-test.csv")
df1["EmbeddingType"] = "ProtTrans emb."
df2 = pd.read_csv("test-nospecies.csv")
df2["EmbeddingType"] = "Adaptive emb."
df = pd.concat([df2, df1])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = 0


cat = "AA"
subcat = "EmbeddingType"
val = "avg AUC ROC"
err = "CI"
title = "Test set - pre-trained LM embedding"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5], tilted=True, legend_outside=False, add_p = True)





df = pd.read_csv("singletests.csv")
cat = "AA"
title = "Optimized model - Cross-validation"
subcat = "Metric"
err = "CI"
val = "avg_metric"

df_seperate = pd.DataFrame(columns = ["AA", "avg_metric", "std_metric", "Metric", "CI"])
for metric in ["AUC ROC", "AUC PR", "MCC"]:
    avg_metric = f"avg {metric}"
    std_metric = f"std {metric}"
    df_metric = df[["AA", avg_metric, std_metric]]
    df_metric.columns = ["AA", "avg_metric", "std_metric"]
    df_metric["Metric"] = avg_metric
    df_seperate = pd.concat([df_seperate, df_metric])
df_seperate["CI"] = (df_seperate["std_metric"]/sqrt(25))

# call the function with df from the question
grouped_barplot(df_seperate, cat, subcat, val, err, title,  [0.5,1], [8.5, 4.5], legend_outside=True)




df1 = pd.read_csv("holdout.csv")
df1["Feature type"] = "Seq+Species (FC)"

df2 = pd.read_csv("noholdout.csv")
df2["Feature type"] = "Seq+Species (Conv)"

df3 = pd.read_csv("singletests.csv")
df3["Feature type"] = "Sequence only"


df1 = df1.set_index('AA')
df1 = df1.reindex(index=df3['AA'])
df1 = df1.reset_index()

df2 = df2.set_index('AA')
df2 = df2.reindex(index=df3['AA'])
df2 = df2.reset_index()


df = pd.concat([df1, df2, df3])
df = df[df.AA.duplicated(keep=False)]


df["CI"] = df["std AUC ROC"]/sqrt(25)

cat = "AA"
subcat = "Feature type"
val = "avg AUC ROC"
err = "CI"
title = "Species feature - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title,  [0.5,1], [8.5, 4.5], legend_outside=True, add_p = True)



df1 = pd.read_csv("test-nospecies.csv")
df1["FeatureType"] = "Test: Seq only"
df2 = pd.read_csv("test-species.csv")
df2["FeatureType"] = "Test: Seq + species"

"""
df3 = pd.read_csv("singletests.csv")
df3["FeatureType"] = "Val: Seq only"
df4 = pd.read_csv("val_species.csv")
df4["FeatureType"] = "Val: Seq + species"
"""

df1 = df1.set_index('AA')
df1 = df1.reindex(index=df2['AA'])
df1 = df1.reset_index()

"""
df4 = df4.set_index('AA')
df4 = df4.reindex(index=df3['AA'])
df4 = df4.reset_index()
"""

df = pd.concat([df1, df2])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = 0

cat = "AA"
subcat = "FeatureType"
val = "avg AUC ROC"
err = "CI"
title = "Test set - Species Feature"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [6.7, 4.5], legend_outside=False, tilted=True)



"""
df1 = pd.read_csv("test-nospecies.csv")
df1["FeatureType"] = "Test: Seq only"
"""

df2 = pd.read_csv("test-species.csv")
df2["FeatureType"] = "Final model: Test"

"""
df3 = pd.read_csv("singletests.csv")
df3["FeatureType"] = "Val: Seq only"
"""

df4 = pd.read_csv("val_species.csv")
df4["FeatureType"] = "Final model: Val"


df4["CI"] = df4["std AUC ROC"]/sqrt(25)
df2["CI"] = 0
df2 = df2.set_index('AA')
df2 = df2.reindex(index=df4['AA'])
df2 = df2.reset_index()

df = pd.concat([df4, df2])
df = df[df.AA.duplicated(keep=False)]


cat = "AA"
subcat = "FeatureType"
val = "avg AUC ROC"
err = "CI"
title = "Validation- and Test set comparison"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [6.7, 4.5], legend_outside=False, tilted=True)




df1 = pd.read_csv("webserver-test.csv")
df1["FeatureType"] = "MusiteDeep model"
df2 = pd.read_csv("test-species-10k-2010.csv")
df2["FeatureType"] = "Our model"


df2 = df2.set_index('AA')
df2 = df2.reindex(index=df1['AA'])
df2 = df2.reset_index()

df = pd.concat([df1, df2])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = 0

cat = "AA"
subcat = "FeatureType"
val = "avg AUC ROC"
err = "CI"
title = "Test set - Musite comparison (2010 split)"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [6.7, 4.5], legend_outside=False, tilted=True)




df1 = pd.read_csv("MusiteTest-them.csv")
df1["FeatureType"] = "Testset 2010 (10k): Musite"
df2 = pd.read_csv("MusiteTest-US.csv")
df2["FeatureType"] = "Testset 2010 (10k): Us"


df2 = df2.set_index('AA')
df2 = df2.reindex(index=df1['AA'])
df2 = df2.reset_index()

df = pd.concat([df1, df2])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = df["std AUC ROC"]/sqrt(25)
cat = "AA"
subcat = "FeatureType"
val = "avg AUC ROC"
err = "CI"
title = "Test set - Musite comparison (10k samples, 2010 split)"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [2.5, 4.5], legend_outside=True, tilted=True)





"""
df = pd.read_csv("MultitaskTest.csv")
df["FeatureType"] = "Test: Multitask"
df["CI"] = (df["std MCC (species)"] * 1.96)/5
df = df[["avg MCC (species)","std MCC (species)", "AA", "FeatureType", "CI"]]

df2 = pd.read_csv("Multitask.csv")
df2["FeatureType"] = "Val: Multitask"
df2["CI"] = (df2["std MCC (species)"] * 1.96)/5
df2 = df2[["avg MCC (species)","std MCC (species)", "AA", "FeatureType", "CI"]]

df2 = df2.set_index('AA')
df2 = df2.reindex(index=df['AA'])
df2 = df2.reset_index()

df = pd.concat([df, df2])
df = df[df.AA.duplicated(keep=False)]

cat = "AA"
subcat = "FeatureType"
val = "avg MCC (species)"
err = "CI"
title = "Test set - species task multitask"

grouped_barplot(df, cat, subcat, val, err, title, [0.0,1], [8.5, 4.5], legend_outside=True, tilted=True)
"""


