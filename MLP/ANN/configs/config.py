config = {
    "files": {
        "data_submission": "../data/sample_submission.csv",
        "data_train": "../data/train.csv",
        "data_test":"../data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": "test_longtime_1500_final"
    },
    "model_params": {
        "use_dropout": False,
        "pred_size":8,
        "tst_size":96,
        "input_size":26,
        "hidden_dim":256
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 128,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.000001},
        "device": "cpu",
        "epochs": 1000,
        "pbar": True,
        "min_delta": 0,
        "patience": 150,
    },
    "train": True,
    "validation": False,
}
