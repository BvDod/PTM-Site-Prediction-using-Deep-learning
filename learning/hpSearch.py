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
    metric = testModel(parameters, trial=trial, logToComet=False, )
    return metric

def evaluateBestTrial(best_params, tuning_settings, parameters):
    parameters["aminoAcid"] = tuning_settings["aminoAcid"]
    for parameter, value in best_params.items():
        parameters[parameter] = value
    
    avg_dict, std_dict = testModel(parameters, logToComet=True, returnEvalMetrics=True)
    return avg_dict, std_dict

    

    
if __name__ == "__main__":

    parameters = { 
        # Training parameters
        "gpu_mode": True,
        "epochs": 100,
        "batch_size": 4096,
        "learning_rate": None,
        "test_data_ratio": 0.1,
        "data_sample_mode": "undersample",
        "loss_function": nn.BCEWithLogitsLoss,
        "optimizer": optim.AdamW,
        "crossValidation": False,
        "folds": 5,
        "earlyStopping": True,
        "ValidationMetric": "Validation Loss",
        "earlyStoppingPatience": 10,
        "CV_Repeats": 1,
        
        # Model parameters
        "weight_decay": None,
        "embeddingType": "adaptiveEmbedding",
        "embeddingSize": 200,
        "CNN_layers": 2,
        "CNN_filters": 200,
        "CNN_dropout": 0,
        "LSTM_layers": 1,
        "LSTM_hidden_size": 128,
        "LSTM_dropout": 0,
        "FC_layers": 2,
        "FC_layer_size": 256,
        "FC_dropout": 0.5, 
        "embeddingDropout": 0,
        }

    tuning_settings = {
        "n_trials": 250,
        "aminoAcid": "Phosphorylation-ST",
        "FloatsToTune" : {
            "learning_rate": [0.00001, 0.01],
            "weight_decay": [0, 100],
            "FC_dropout": [0.4, 0.75],
            "CNN_dropout": [0, 0.5],
            "embeddingDropout": [0,0.5]
        },
        "IntsToTune" : {
        }
    }

    """
    trial_params = {
        "CNN_dropout": 	0.20755,
        "CNN_layers": 1,
        "FC_dropout": 0.69120,
        "learning_rate": 0.003885,
        "weight_decay": 5.53029,
        "embeddingSize": 178,
        "embeddingDropout": 0.449
    }

    bestEvalMetricsAvg, bestEvalMetricsSTD = evaluateBestTrial(trial_params, tuning_settings, parameters)
    print(bestEvalMetricsAvg)
    exit()
    """

    optunaObjective = lambda trial: objective(trial, tuning_settings, parameters)      
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner(), sampler=optuna.samplers.TPESampler(multivariate=True,))
    study.optimize(optunaObjective, n_trials=tuning_settings["n_trials"], gc_after_trial=True)

    bestEvalMetricsAvg, bestEvalMetricsSTD = evaluateBestTrial(study.best_trial.params, tuning_settings, parameters)
    experiment = createHpTuningLogger(tuning_settings, parameters)
    logHpStudy(study, experiment, bestEvalMetricsAvg, bestEvalMetricsSTD)


