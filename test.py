if 3:
    print("test")

aminoAcids = {
    }


# SINGLE MODELS
aminoAcids = {
        "Hydroxylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0015666979 ,
            "weight_decay": 23.27890382,
            "CV_Repeats":5,
            "crossValidation": True},
        "Hydroxylation-P": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.002079141428,
            "weight_decay": 7.1738,
            "CV_Repeats":5,
            "crossValidation": True},
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
            "CV_Repeats":5,
            "crossValidation": True},
        "Sumoylation": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0009745544177,
            "weight_decay": 0.5650471043,
            "CV_Repeats":5,
            "crossValidation": True},
        "S-palmitoylation-C": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 5.87E-04,
            "weight_decay": 1.950457673,
            "CV_Repeats":5,
            "crossValidation": True},
        "Methylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 25,
            "learning_rate": 5.36E-04,
            "weight_decay": 1.02323669,
            "CV_Repeats":5,
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
            "crossValidation": True},
    }

# Single model +
aminoAcids = {
        "Hydroxylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0015666979 ,
            "weight_decay": 23.27890382,
            "CV_Repeats":5,
            "crossValidation": True},
        "Hydroxylation-P": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.002079141428,
            "weight_decay": 7.1738,
            "CV_Repeats":5,
            "crossValidation": True},
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
            "CV_Repeats":5,
            "crossValidation": True},
        "Sumoylation": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 0.0009745544177,
            "weight_decay": 0.5650471043,
            "CV_Repeats":5,
            "crossValidation": True},
        "S-palmitoylation-C": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 50,
            "learning_rate": 5.87E-04,
            "weight_decay": 1.950457673,
            "CV_Repeats":5,
            "crossValidation": True},
        "Methylation-K": {
            "data_sample_mode": ["oversample",],
            "earlyStoppingPatience": 25,
            "learning_rate": 5.36E-04,
            "weight_decay": 1.02323669,
            "CV_Repeats":5,
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
            "crossValidation": True},
    }


"""
"Phosphorylation-Y": {
{'learning_rate': 0.0003671427201005768, 'weight_decay': 0.204501061010248}

['Pyrrolidone carboxylic acid']
learning_rate           : 0.0033413418609678004
weight_decay            : 3.2909420333866857

"Hydroxylation-P"
{'learning_rate': 0.002079141427786005, 'weight_decay': 7.173835966047036}
"""