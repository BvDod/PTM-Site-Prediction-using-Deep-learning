# %%
import os
from re import L
import numpy
import pandas as pd
import csv
from functions.EvaluationMetrics import get_evaluation_metrics
import numpy as np
import pandas as pd


def get_species_names(df):
    df["Mnemonic"] = df.UniprotID.str.split("_").str[-1]
    df_mm = pd.read_csv(f"data/allmm.csv", delimiter="\t")

    df_mm.loc[df_mm["Common name"].isnull(), "Common name"] = df_mm.loc[df_mm["Common name"].isnull(), "Scientific name"]

    df = pd.merge(df, df_mm[["Mnemonic", "Common name"]], on="Mnemonic", how='left')
    df["Common name"] = df["Common name"].fillna("Unknown")
    df["Common name"] = df["Common name"].astype("string")
    return df

input_dir = "data/MusiteTest2010_static"
input_dir = "data/MusiteTest_musiteowndata_static"
# %%

species_labels = pd.read_csv("SpeciesLabels.csv", index_col=0, header=0, squeeze=True).to_dict()
for file in os.listdir(input_dir):
    AA = file
    print(AA)
    if not os.path.isdir(f"{input_dir}/{file}/"):
        continue      
    dir = f"{input_dir}/{file}"

    df = pd.read_csv(f"{dir}/df_test")
    print(len(df))
    df = get_species_names(df)


    species = []
    for index, series in df.iterrows():
        if series["Common name"] in species_labels:
            species.append(species_labels[series["Common name"]])
        else:
            species.append(species_labels["other"])
    species = np.array(species)

    with open(f"{dir}/labels.txt", "r") as file:
        lines = file.readlines()
        labels = [int(line.rstrip()) for line in lines]

    predictions = []
    with open(f"{dir}/Prediction_results.txt", "r") as file:
        
        csv_reader = csv.reader(file, delimiter='\t')
        
        has_pred = True
        for row in csv_reader:

            if row[0] == "ID":
                continue
                
            if row[0][0] == ">":
                if has_pred == False:
                    # print(f"error: no pred for seq: {row[0][1:]}")
                    predictions.append(0)
                
                if len(predictions) != int(row[0][1:]):
                    predictions.append(0)

                has_pred = False
                continue
            
            if int(row[1]) == 17:
                has_pred = True
                predictions.append(float(row[3].split(":")[-1]))
        if has_pred == False:
                    # print(f"error: no pred for seq: {row[0][1:]}")
                    predictions.append(0)

    y_true = np.array(labels)
    y_output = np.array(predictions)
    y_pred = y_output > 0.5

    specie_metrics = True
    eval_metrics, eval_figures = get_evaluation_metrics(AA, y_true, y_output, y_pred, figures=False, species=False) 
    print(f'{eval_metrics[f"{AA} AUC ROC"]}, {eval_metrics[f"{AA} AUC PR"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} Validation Sensitivity"]}, {eval_metrics[f"{AA} Validation Specificity"]}, {eval_metrics[f"{AA} Validation Precision"]}')

    specie_metrics = True
    if specie_metrics:
        for specie in np.unique(species):
            print(specie)
            print(sum(species == specie))
            y_true_species = y_true[species == specie]
            y_output_species = y_output[species == specie]
            y_pred_species = y_pred[species == specie]
            eval_metrics, eval_figures = get_evaluation_metrics(AA, y_true_species, y_output_species, y_pred_species, figures=False, species=False) 
            print(f'{eval_metrics[f"{AA} AUC ROC"]}, {eval_metrics[f"{AA} AUC PR"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} Validation Sensitivity"]}, {eval_metrics[f"{AA} Validation Specificity"]}, {eval_metrics[f"{AA} Validation Precision"]}, {sum(species == specie)}')



    print("Baseline random")
    import random
    y_output = numpy.random.uniform(size=y_output.size)
    y_pred = y_output > 0.5
    eval_metrics, eval_figures = get_evaluation_metrics(AA, y_true, y_output, y_pred, figures=False, species=False) 
    print(f'{eval_metrics[f"{AA} AUC ROC"]}, {eval_metrics[f"{AA} AUC PR"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} Validation Sensitivity"]}, {eval_metrics[f"{AA} Validation Specificity"]}, {eval_metrics[f"{AA} Validation Precision"]}')

    print("Baseline majority class")
    y_output = np.full_like(y_output, 0)
    y_pred = y_output > 0.5
    eval_metrics, eval_figures = get_evaluation_metrics(AA, y_true, y_output, y_pred, figures=False, species=False) 
    print(f'{eval_metrics[f"{AA} AUC ROC"]}, {eval_metrics[f"{AA} AUC PR"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} Validation Sensitivity"]}, {eval_metrics[f"{AA} Validation Specificity"]}, {eval_metrics[f"{AA} Validation Precision"]}')


    # print(eval_metrics)
    print()

    """
    if len(labels) < 1000:
        continue
    df = pd.read_csv(f"{dir}/df_test")
    df["DateSeqModified"] = pd.to_datetime(df["DateSeqModified"])
    year_dict = {}
    for year in sorted(list(df.DateSeqModified.dt.year.unique())):
        y_true_year = []
        y_output_year = []
        y_pred_year = []
        for i, (index, row) in enumerate(df.iterrows()):
            if row["DateSeqModified"].year == year:
                y_true_year.append(labels[i])
                y_output_year.append(predictions[i])
                y_pred_year.append(y_pred[i])
        if (len(y_true) < 200) or (sum(y_true) < ):
            continue
        year_dict[str(year)] = [np.array(y_true_year), np.array(y_output_year), np.array(y_pred_year)]
    
    for year, result_list in year_dict.items():
        print(year)
        eval_metrics, eval_figures = get_evaluation_metrics(AA, result_list[0], result_list[1], result_list[2], figures=False, species=False)
        print(f'{eval_metrics[f"{AA} AUC ROC"]}, {eval_metrics[f"{AA} AUC PR"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} MCC"]}, {eval_metrics[f"{AA} Validation Sensitivity"]}, {eval_metrics[f"{AA} Validation Specificity"]}, {eval_metrics[f"{AA} Validation Precision"]}')
        print()
    """
        

