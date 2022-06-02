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
    metric = testModel(parameters, trial=trial, logToComet=False, device_id=3
    return metric

def evaluateBestTrial(best_params, tuning_settings, parameters):
    parameters["aminoAcid"] = tuning_settings["aminoAcid"]
    for parameter, value in best_params.items():
        parameters[parameter] = value
    parameters["CV_Repeats"] = 5
    
    avg_dict, std_dict = testModel(parameters, logToComet=True, returnEvalMetrics=True)
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
    for param, value in tuning_settings.items():
        parameters[param] = value

    optunaObjective = lambda trial: objective(trial, tuning_settings, parameters)      
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner(), sampler=optuna.samplers.TPESampler(multivariate=True,))
    study.optimize(optunaObjective, n_trials=tuning_settings["n_trials"], gc_after_trial=True)

    bestEvalMetricsAvg, bestEvalMetricsSTD = evaluateBestTrial(study.best_trial.params, tuning_settings, parameters)
    experiment = createHpTuningLogger(tuning_settings, parameters)
    logHpStudy(study, experiment, bestEvalMetricsAvg, bestEvalMetricsSTD)

    
if __name__ == "__main__":

    parameters = { 
        # Training parameters
        "gpu_mode": True,
        "epochs": 200,
        "batch_size": 2048,
        "learning_rate": 0.0005,
        "test_data_ratio": 0.2,
        "data_sample_mode": "oversample",
        "crossValidation": False,
        "loss_function": nn.BCELoss,
        "optimizer": optim.AdamW,
        "folds": 5,
        "earlyStopping": True,
        "ValidationMetric": "Validation Loss (Sumoylation)",
        # "ValidationMetric": "Validation Loss (total)",
        "earlyStoppingPatience": 50,
        "CV_Repeats": 1,
        "Experiment Name": "Multitask - specific AA's test",
        'CreateFigures': False,


        # Model parameters
        "weight_decay": 2.5,
        "embeddingType": "adaptiveEmbedding",
        "LSTM_layers": 1,
        "LSTM_hidden_size": 32,
        "LSTM_dropout": 0,
        "MultiTask": True,

        "MultiTask_sample_method": "balanced",
        "UseUncertaintyBasedLoss": False,
        "useLrWeight": False,

        "CNNType": "Musite",
        "FCType": "Adapt",

        "layerToSplitOn": "FC"
        }
                      

    parameters["data_sample_mode"] = ["oversample"] * 13

    tuning_settings = {
        "aminoAcid" : ["Hydroxylation-K", "Hydroxylation-P", "Pyrrolidone carboxylic acid", "S-palmitoylation-C", "Sumoylation"],
        "n_trials": 250,
        "FloatsToTune" : {
            "learning_rate": [0.00001, 0.01],
            "weight_decay": [0, 25],
        },
        "IntsToTune" : {   
        },
        "crossValidation": True,
        "earlyStoppingPatience": 20,
        "CV_Repeats":1,

    }

    performTuningExperiment(parameters, tuning_settings)






