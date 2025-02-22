from lib2to3.pygram import python_grammar_no_print_statement
from train import testModel
import torch.nn as nn
import torch.optim as optim

from functions.EvaluationMetrics import createHpTuningLogger
from functions.EvaluationMetrics import logHpStudy

import optuna

def defineHyperparameters(trial, tuning_settings, parameters):
        parameters["aminoAcid"] = tuning_settings["aminoAcid"]
        for parameter, (lower, upper) in tuning_settings["FloatsToTune"].items():
            parameters[parameter] = trial.suggest_float(parameter, lower, upper)
        
        for parameter, (lower, upper) in tuning_settings["IntsToTune"].items():
            parameters[parameter] = trial.suggest_int(parameter, lower, upper)

        return parameters


def objective(trial, tuning_settings, parameters):
    parameters = defineHyperparameters(trial, tuning_settings, parameters)
    metric = testModel(parameters, trial=trial, logToComet=False,)
    return metric

def evaluateBestTrial(parameters):
    avg_dict, std_dict = testModel(parameters, logToComet=True, returnEvalMetrics=True, device_id=0)
    return avg_dict, std_dict
    
def performTuningExperiment(parameters, tuning_settings):

    """
    trial_params = {
        "learning_rate": 0.00196,
        "weight_decay": 3,
        "embeddingDropout": 0,
        "FC_dropout": 0.5,
        "CNN_dropout": 0.75,
    }

    bestEvalMetricsAvg, bestEvalMetricsSTD = evaluateBestTrial(trial_params, tuning_settings, parameters)
    print(bestEvalMetricsAvg)
    exit()
    """
    
    optunaObjective = lambda trial: objective(trial, tuning_settings, parameters)      
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner(), sampler=optuna.samplers.TPESampler(multivariate=True,))
    study.optimize(optunaObjective, n_trials=tuning_settings["n_trials"], gc_after_trial=True)

    for name, value in study.best_trial.params.items():
        parameters[name] = value
    bestEvalMetricsAvg, bestEvalMetricsSTD = evaluateBestTrial(parameters)
    experiment = createHpTuningLogger(tuning_settings, parameters)
    logHpStudy(study, experiment, bestEvalMetricsAvg, bestEvalMetricsSTD)

    
if __name__ == "__main__":

    parameters = { 
        # Training parameters
        "gpu_mode": True,
        "epochs": 400,
        "batch_size": 512,
        "learning_rate": None,
        "test_data_ratio": 0.2,
        "data_sample_mode": "balanced",
        "crossValidation": True,
        "loss_function": nn.BCELoss,
        "optimizer": optim.AdamW,
        "folds": 5,
        "earlyStopping": True,
        "ValidationMetric": "Validation Loss (total)",
        "earlyStoppingPatience": 25,
        "CV_Repeats": 1,
        "Experiment Name": "retestt",
        # Model parameters
        "weight_decay": None,
        "embeddingType": "adaptiveEmbedding",
        "LSTM_layers": 1,
        "LSTM_hidden_size": 32,
        "LSTM_dropout": 0,
        "UseUncertaintyBasedLoss": False,
        "useLrWeight": False,
        "CreateFigures": False,
        'SeperateTuningLRandWD': False,
        'dontAverageLoss': False,

        "predictSpecies": False,
        "onlyPredictHumans": False,
        "useSpecieFeature": True,
        "SpecieFeatureHoldout": False,

        "MultiTask_Species": False,
        "species_weight": 0.15,
        "TestSet": True,
        "MusiteTest": True,
        "split_2010": True,
        "specieMetrics": True,
        }

    tuning_settings = {
        "n_trials": 250,
        "aminoAcid": "Methylation-K",
        "FloatsToTune" : {
            "learning_rate": [0.00001, 0.01],
            "weight_decay": [0, 25],
        },
        "IntsToTune" : {   
        },
    }


    aminoAcids = {
        "Phosphorylation-['S', 'T']": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 20,
            "learning_rate": 0.00036135,
            "weight_decay": 0.0961727,
            "CV_Repeats":1,
            "crossValidation": False},
    }
    
    {
        "Phosphorylation-Y": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "learning_rate": 1.98E-04,
            "weight_decay": 0.161356466,
            "CV_Repeats":1,
            "crossValidation": False},
    }
    
    {
        "Hydroxylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0015666979 ,
            "weight_decay": 23.27890382,
            "CV_Repeats":5,
            "crossValidation": True,
            "SpecieFeatureHoldout": True,},
        "Hydroxylation-P": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.002079141428,
            "weight_decay": 7.1738,
            "CV_Repeats":5,
            "crossValidation": True,
            "SpecieFeatureHoldout": True,},
        "Pyrrolidone carboxylic acid": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.00334134,
            "weight_decay": 3.290942,
            "CV_Repeats":5,
            "crossValidation": True},
        "Methylation-R": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 6.91E-04,
            "weight_decay": 3.1783,
            "CV_Repeats":3,
            "crossValidation": True},
        "Sumoylation": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0009745544177,
            "weight_decay": 0.5650471043,
            "CV_Repeats":3,
            "crossValidation": True},
        "S-palmitoylation-C": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 5.87E-04,
            "weight_decay": 1.950457673,
            "CV_Repeats":3,
            "crossValidation": True},
        "Methylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 25,
            "learning_rate": 5.36E-04,
            "weight_decay": 1.02323669,
            "CV_Repeats":3,
            "crossValidation": True},
        "O-linked Glycosylation": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "learning_rate": 0.003566031739,
            "weight_decay": 1.162598511,
            "CV_Repeats":3,
            "crossValidation": True},
        "N-linked Glycosylation": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "learning_rate": 0.001754199373,
            "weight_decay": 0.5873957722,
            "CV_Repeats":3,
            "crossValidation": True},
        "Acetylation": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "learning_rate": 5.92E-04,
            "weight_decay": 0.2448830361,
            "CV_Repeats":1,
            "crossValidation": True},
        "Phosphorylation-Y": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "learning_rate": 1.98E-04,
            "weight_decay": 0.161356466,
            "CV_Repeats":1,
            "crossValidation": True},
        "Ubiquitination": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 20,
            "learning_rate": 5.29E-04,
            "weight_decay": 1.031953666,
            "CV_Repeats":1,
            "crossValidation": True},
        "Phosphorylation-['S', 'T']": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 20,
            "learning_rate": 0.00036135,
            "weight_decay": 0.0961727,
            "CV_Repeats":1,
            "crossValidation": True}, }

    


    """
    for amino_acid, aa_parameters in aminoAcids.items():
        for key, value in aa_parameters.items():
            parameters[key] = value
        parameters["aminoAcid"] = [amino_acid,]
        parameters["CNNType"] = "Adapt"
        parameters["FCType"] = "Musite"
        avg_dict, std_dict = evaluateBestTrial(parameters)
        print(avg_dict, std_dict)
    """

    for CNNType in ["Musite"]:
        for FCType in ["Adapt"]:
            for amino_acid, aa_parameters in aminoAcids.items():
                parameters["CNNType"] = CNNType
                parameters["FCType"] = FCType
                tuning_settings["aminoAcid"] = [amino_acid,]
                parameters["aminoAcid"] = [amino_acid,]
                for key, value in aa_parameters.items():
                    parameters[key] = value
                
                #performTuningExperiment(parameters, tuning_settings)
                evaluateBestTrial(parameters)







