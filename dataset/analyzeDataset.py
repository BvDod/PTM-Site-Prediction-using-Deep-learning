# %%
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# %%

def get_merged_df(input_dir):
    dataframes = []
    for file in os.listdir(input_dir):
        df = pd.read_csv(f"{input_dir}/{file}", delimiter=",")


        if file.split("_")[-4] == "neg":
            df["label"] = 0
        elif file.split("_")[-4] == "pos":
            df["label"] = 1
            df = df.drop(columns=["truncateStatus","TruncatedUniProtSequence"])
        else:
            print("Error: invalid filename")
            exit()

        df["PTM_type"] = file.split("_")[0]
        dataframes.append(df)

    df_all = pd.concat(dataframes)
    return dataframes, df_all

input_dir = "data/learningData/balanced/final_split/train/"
dfs, df_all = get_merged_df(input_dir)

# input_dir = "data/processed/final_split/test"
# dfs, df_all_test = get_merged_df(input_dir)



# %%
import statistics

def print_stats(df):
    pos_n = len(df[df.label == 1])
    print(f"Positive samples: {pos_n}")
    neg_n = len(df[df.label == 0])
    print(f"Negative samples: {neg_n}\n")

    ptms_amount = len(df["PTM_type"].unique())
    print(f"Unique PTM types: {ptms_amount}")
    unique_proteins = len(df["UniprotAC"].unique())
    print(f"Unique proteins: {unique_proteins}\n")

    counts = df["UniprotAC"].value_counts().tolist()
    print(f"Samples per protein - Mean: {statistics.mean(counts)}")
    print(f"Samples per protein - Median: {statistics.median(counts)}")
    print(f"Samples per protein - Maximum: {max(counts)}\n\n")

print("Train dataset")
print_stats(df_all)

print("Test dataset")
# print_stats(df_all_test)







# %%

def get_species_names(df):
    df["Mnemonic"] = df.UniprotID.str.split("_").str[-1]
    df_mm = pd.read_csv(f"data/allmm.csv", delimiter="\t")

    df_mm.loc[df_mm["Common name"].isnull(), "Common name"] = df_mm.loc[df_mm["Common name"].isnull(), "Scientific name"]

    df = pd.merge(df, df_mm[["Mnemonic", "Common name"]], on="Mnemonic", how='left')
    df["Common name"] = df["Common name"].fillna("Unknown")
    df["Common name"] = df["Common name"].astype("string")
    return df

df_all = get_species_names(df_all)
# df_all_test = get_species_names(df_all_test)

dfs = [get_species_names(df) for df in dfs]



# %%

def plot_species_count(df, string):

    print(len(df["Common name"].unique()))
    for index, row in df.iterrows():
        if len(row["Common name"]) > 15:
            row["Common name"] = row["Common name"][:14] + "."
            df.at[index,"Common name"] = row["Common name"]

    species_count = df["Common name"].value_counts().to_dict()
    x, y = zip(*species_count.items())

    sns.color_palette("cubehelix")
    sns.set()
    max_species = 15

    x_short = list(x[:max_species])
    y_short = list(y[:max_species])
    x_short.append("Other")
    y_short.append(sum(y[max_species:]))

    x_short = [string.split("(")[0] for string in x_short]
    patches, texts = plt.pie(y, startangle=90, radius=1.2)
    percent = [100.* value/sum(y) for value in y_short]
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x_short, percent)]

    plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.title(f"{string} Dataset - Species", loc="right", fontsize=18)
    plt.savefig(f'species_{string}.png', dpi=300, bbox_inches='tight')

plot_species_count(df_all, "Train")

# plot_species_count(df_all_test, "Test")

for df in dfs:
    plot_species_count(df, df["PTM_type"].iloc[0])



# %%
import matplotlib.dates as mdates


def plot_date_hist(df, string, ax= None):
    dates = df["DateSeqModified"].to_list()
    dates_mpl = mdates.datestr2num(dates)

    sns.set()
    if ax == None:
        fig, ax = plt.subplots(1,1)
    ax.hist(dates_mpl, bins=35)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.title(f"{string} Dataset - Year of sequencing")
    plt.xlabel('Date')
    plt.ylabel('Frequency')    
    plt.savefig(f'year_{string}.png', dpi=300, bbox_inches='tight')
    return ax

def plot_date_hist_overlapping(df, df_test, ax= None):
    dates = df["DateSeqModified"].to_list()
    dates_mpl = mdates.datestr2num(dates)

    dates_test = df_test["DateSeqModified"].to_list()
    dates_mpl_test = mdates.datestr2num(dates_test)

    sns.set()
    fig, ax = plt.subplots(1,1)

    ax.hist([dates_mpl, dates_mpl_test], bins=35)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.title(f"Dataset - Year of sequencing")
    plt.legend(["Train", "Test"])
    plt.savefig(f'year_merged.png', dpi=300, bbox_inches='tight')
    return ax


ax = plot_date_hist(df_all, "Train")

# plot_date_hist_overlapping(df_all, df_all_test)


# %%

def plot_modified_AAs(df, string):
    sns.color_palette("cubehelix")
    sns.set()

    fig, axes = plt.subplots(4, 4, figsize=(18,15))
    axes = axes.flatten()

    for i, AA in enumerate(df["PTM_type"].unique()):
        ax = axes[i]
        AA_count = df.loc[df["PTM_type"] == AA, "ModifiedAA"].value_counts().to_dict()
        x, y = zip(*AA_count.items())

        patches, texts = ax.pie(y, startangle=90, radius=1.2)
        percent = [100.* value/sum(y) for value in y]
        labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]

        ax.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.set_title(AA)

    for ax in axes[len(df["PTM_type"].unique()):]:
        ax.set_visible(False)
        
    df.loc[df["PTM_type"] == AA, "ModifiedAA"].value_counts()
    fig.suptitle(f"{string} dataset - Amino acids modified by PTM", fontsize=35)
    plt.savefig(f'modifiedAAs_{string}.png', dpi=300, bbox_inches='tight')
plot_modified_AAs(df_all, "Train")
plot_modified_AAs(df_all_test, "Test")

# %%


def plot_sample_counts(df, string):
    sns.color_palette("cubehelix")
    sns.set()

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_autopct


    fig, axes = plt.subplots(4, 4, figsize=(18,15))
    axes = axes.flatten()

    for i, AA in enumerate(df["PTM_type"].unique()):
        ax = axes[i]
        AA_count = df.loc[df["PTM_type"] == AA, "label"].value_counts().to_dict()
        x, y = zip(*AA_count.items())

        ax.pie(y, labels = x, startangle=90, radius=1.2, autopct=make_autopct(y))
        ax.set_title(AA)

    for ax in axes[len(df["PTM_type"].unique()):]:
        ax.set_visible(False)

    fig.suptitle(f"{string}Dataset - Sample Counts", fontsize=35)
    plt.savefig(f'SampleCounts_{string}.png', dpi=300, bbox_inches='tight')
plot_sample_counts(df_all, "Train")
plot_sample_counts(df_all_test, "Test")


# %%

def plot_sample_pies(df, string):
    counts = df["UniprotAC"].value_counts().tolist()
    plt.hist(counts, range=[0,125], bins=25)
    plt.title(f"{string} dataset - PTM samples per Protein")
    plt.xlabel("Samples per protein")
    plt.ylabel("Frequency")
    plt.text(85, 1300, f'Maximum: {counts[0]}', fontsize = 14)
    plt.savefig(f'SamplesPerProtein_{string}.png', dpi=300, bbox_inches='tight')
    plt.show()
plot_sample_pies(df_all, "Train")
plot_sample_pies(df_all_test, "Test")

# %%
