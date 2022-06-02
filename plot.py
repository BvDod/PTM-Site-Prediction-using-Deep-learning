import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from math import sqrt


df =  pd.read_csv("embedding.csv")
df["N"] = 25
df["CI"] = (df["std AUC ROC"] * 1.96) / 5

sns.set_theme()
sns.set_style("ticks")
sns.set_palette("deep")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


save = True

def grouped_barplot(df, cat,subcat, val , err, title, range, figsize, tilted=True, legend_outside=False):
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

    
    plt.ylim(*range)

    if legend_outside:
        plt.legend(loc='upper left', bbox_to_anchor=(1.025, 1.0))
    else:
        plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig(f"figures/{title}.png", bbox_inches = "tight", dpi=300)
    else:
        plt.show()




cat = "AA"
subcat = "EmbeddingType"
val = "avg  AUC ROC"
err = "CI"
title = "AA representation - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5])






df =  pd.read_csv("sampling.csv")
df["CI"] = (df["std AUC ROC"] * 1.96) / 5
cat = "AA"
subcat = "Sampling Method"
val = "avg  AUC ROC"
err = "CI"
title = "Sampling Method - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5])






df =  pd.read_csv("arch2.csv")
df["CI"] = (df["std AUC ROC"] * 1.96) / 5
print(df)
cat = "AA"
subcat = "Architecture"
val = "avg  AUC ROC"
err = "CI"
title = "Model Architecture - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [4, 4.5])






df1 = pd.read_csv("protBert.csv")
df1["EmbeddingType"] = "ProtBert"
df2 = pd.read_csv("singletests.csv")
df2["EmbeddingType"] = "adaptiveEmbedding"
df = pd.concat([df1, df2])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = (df["std AUC ROC"] * 1.96) / 5

cat = "AA"
subcat = "EmbeddingType"
val = "avg MCC"
err = "CI"
title = "ProtBert Embeddings - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0,1], [5, 4.5], tilted=True)






df = pd.read_csv("singletests.csv")
cat = "AA"
title = "Single model tests"
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
df_seperate["CI"] = (df_seperate["std_metric"] * 1.96) / 5
print(df_seperate)

# call the function with df from the question
grouped_barplot(df_seperate, cat, subcat, val, err, title,  [0,1], [8.5, 4.5], legend_outside=True)





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

print(df1)
print(df2)
print(df3)

df = pd.concat([df1, df2, df3])
df = df[df.AA.duplicated(keep=False)]


df["CI"] = (df["std AUC ROC"] * 1.96) / 5

cat = "AA"
subcat = "Feature type"
val = "avg AUC ROC"
err = "CI"
title = "Species feature - Experiment"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title,  [0.5,1], [8.5, 4.5], legend_outside=True)




df1 = pd.read_csv("test-nospecies.csv")
df1["FeatureType"] = "Test: Seq only"
df2 = pd.read_csv("test-species.csv")
df2["FeatureType"] = "Test: Seq + species"
df3 = pd.read_csv("singletests.csv")
df3["FeatureType"] = "Val: Seq only"

df1 = df1.set_index('AA')
df1 = df1.reindex(index=df3['AA'])
df1 = df1.reset_index()

df2 = df2.set_index('AA')
df2 = df2.reindex(index=df3['AA'])
df2 = df2.reset_index()

df = pd.concat([df1, df2, df3])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = (df["std MCC"] * 1.96) / 5

print(df)
cat = "AA"
subcat = "FeatureType"
val = "avg MCC"
err = "CI"
title = "Test set"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0,1], [8.5, 4.5], legend_outside=True, tilted=True)





df1 = pd.read_csv("test-nospecies.csv")
df1["FeatureType"] = "Test: Seq only"
df2 = pd.read_csv("test-species.csv")
df2["FeatureType"] = "Test: Seq + species"
df3 = pd.read_csv("MultitaskTest.csv")
df3["FeatureType"] = "Test: Multitask"
print(df3)
df3 = df3.drop(columns = ["avg MCC (species)","std MCC (species)"])

df1 = df1.set_index('AA')
df1 = df1.reindex(index=df3['AA'])
df1 = df1.reset_index()

df2 = df2.set_index('AA')
df2 = df2.reindex(index=df3['AA'])
df2 = df2.reset_index()

df = pd.concat([df1, df2, df3])
df = df[df.AA.duplicated(keep=False)]

df["CI"] = (df["std AUC ROC"])

print(df)
cat = "AA"
subcat = "FeatureType"
val = "avg AUC ROC"
err = "CI"
title = "Test set - multitask"
# call the function with df from the question
grouped_barplot(df, cat, subcat, val, err, title, [0.5,1], [8.5, 4.5], legend_outside=True, tilted=True)


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



