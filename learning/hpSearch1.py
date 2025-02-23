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
    metric = testModel(parameters, trial=trial, logToComet=False, device_id=0)
    return metric

def evaluateBestTrial(parameters):
    parameters["crossValidation"] = True
    parameters["CV_Repeats"] = 5
    
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
        "epochs": 200,
        "batch_size": 512,
        "learning_rate": None,
        "test_data_ratio": 0.2,
        "data_sample_mode": "oversample",
        "crossValidation": True,
        "loss_function": nn.BCELoss,
        "optimizer": optim.AdamW,
        "folds": 5,
        "earlyStopping": True,
        "ValidationMetric": "Validation Loss (total)",
        "earlyStoppingPatience": 50,
        "CV_Repeats": 1,
        "Experiment Name": "Single model tuning and eval: ",
        # Model parameters
        "weight_decay": None,
        "embeddingType": "adaptiveEmbedding",
        "LSTM_layers": 1,
        "LSTM_hidden_size": 32,
        "LSTM_dropout": 0,
        "UseUncertaintyBasedLoss": False,
        "useLrWeight": False,
        "CNNType": "Musite",
        "FCType": "Adapt",
        "CreateFigures": False,
        }


    tuning_settings = {
        "n_trials": 250,
        "aminoAcid": "O-linked Glycosylation",
        "FloatsToTune" : {
            "learning_rate": [0.00001, 0.01],
            "weight_decay": [0, 25],
        },
        "IntsToTune" : {   
        },
    }

    aminoAcids = {
        "Phosphorylation-Y": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "CV_Repeats":1,
            "crossValidation": False,},
        "O-linked Glycosylation": {
            "data_sample_mode": ["balanced",],
            "earlyStoppingPatience": 25,
            "CV_Repeats":1,
            "crossValidation": True,}
    }

    for CNNType in ["Musite"]:
        for FCType in ["Adapt"]:
            for amino_acid, aa_parameters in aminoAcids.items():
                parameters["CNNType"] = CNNType
                parameters["FCType"] = FCType
                tuning_settings["aminoAcid"] = [amino_acid,]
                parameters["aminoAcid"] = [amino_acid,]
                for key, value in aa_parameters.items():
                    parameters[key] = value
                performTuningExperiment(parameters, tuning_settings)



