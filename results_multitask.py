{'CNNType': 'Adapt',
 'CV_Repeats': 1,
 'Experiment Name': 'Model architecture - added max, ranges, bceloss',
 'FCType': 'Musite',
 'LSTM_dropout': 0,
 'LSTM_hidden_size': 32,
 'LSTM_layers': 1,
 'MultiTask': True,
 'MultiTask_sample_method': 'balanced',
 'UseUncertaintyBasedLoss': False,
 'ValidationMetric': 'Validation Loss (total)',
 'aminoAcid': ['Hydroxylation-K',
               'Hydroxylation-P',
               'Pyrrolidone carboxylic acid',
               'S-palmitoylation-C',
               'Sumoylation'],
 'batch_size': 2048,
 'crossValidation': False,
 'data_sample_mode': ['oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample',
                      'oversample'],
 'earlyStopping': True,
 'earlyStoppingPatience': 50,
 'embeddingType': 'adaptiveEmbedding',
 'epochs': 200,
 'folds': 5,
 'gpu_mode': True,
 'layerToSplitOn': 'FC',
 'learning_rate': 0.0005,
 'loss_function': <class 'torch.nn.modules.loss.BCELoss'>,
 'optimizer': <class 'torch.optim.adamw.AdamW'>,
 'random_state': 2458807009,
 'test_data_ratio': 0.2,
 'useLrWeight': False,
 'weight_decay': 2.5}
(48, 33)
(190, 33)
Loaded folder code/Thesis/dataset/train/Hydroxylation-K/indices (238 samples)
(176, 33)
(819, 33)
Loaded folder code/Thesis/dataset/train/Hydroxylation-P/indices (995 samples)
(237, 33)
(1306, 33)
Loaded folder code/Thesis/dataset/train/Pyrrolidone carboxylic acid/indices (1543 samples)
(2525, 33)
(10073, 33)
Loaded folder code/Thesis/dataset/train/S-palmitoylation-C/indices (12598 samples)
(3241, 33)
(20251, 33)
Loaded folder code/Thesis/dataset/train/Sumoylation/indices (23492 samples)
12 305
52 1325
82 2090
633 16139
1271 32406
[1,     1] loss: 3.516
[2,    26] loss: 3.273
[3,    26] loss: 2.787
[4,    26] loss: 2.027
[5,    26] loss: 1.574
[6,    26] loss: 1.374
[7,    26] loss: 1.202
[8,    26] loss: 1.054
[9,    26] loss: 0.915
[10,    26] loss: 0.773
[11,    26] loss: 0.650
[12,    26] loss: 0.545
[13,    26] loss: 0.439
[14,    26] loss: 0.368
[15,    26] loss: 0.312
[16,    26] loss: 0.293
[17,    26] loss: 0.291
[18,    26] loss: 0.216
[19,    26] loss: 0.298
[20,    26] loss: 0.331
[21,    26] loss: 0.259
[22,    26] loss: 0.186
[23,    26] loss: 0.155
[24,    26] loss: 0.142
[25,    26] loss: 0.275
[26,    26] loss: 0.279
[27,    26] loss: 0.169
[28,    26] loss: 0.133
[29,    26] loss: 0.121
[30,    26] loss: 0.117
[31,    26] loss: 0.139
[32,    26] loss: 0.133
[33,    26] loss: 0.298
[34,    26] loss: 0.773
[35,    26] loss: 0.546
[36,    26] loss: 0.278
[37,    26] loss: 0.189
[38,    26] loss: 0.158
[39,    26] loss: 0.201
[40,    26] loss: 0.201
[41,    26] loss: 0.281
[42,    26] loss: 0.207
[43,    26] loss: 0.177
[44,    26] loss: 0.147
[45,    26] loss: 0.118
[46,    26] loss: 0.125
[47,    26] loss: 0.134
[48,    26] loss: 0.210
[49,    26] loss: 0.820
[50,    26] loss: 0.619
[51,    26] loss: 0.357
[52,    26] loss: 0.216
Early stopping applied (best metric=2.553823709487915)
Finished Training
Total time taken: 103.87415862083435
