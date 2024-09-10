from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from pyassessors.errors import compute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import datasets

import torch
import neural
import torch.nn as nn

import json
import os

def get_closest_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))

def get_layers(n):
    n = get_closest_power_of_two(n)
    layers = [n, 2*n]
    if n <= 32:
        layers.append(16)
    else:
        while n > 32:
            n //= 2
            layers.append(n)
    return layers

device = "cuda" if torch.cuda.is_available() else "cpu"

ERRORS = ["L1s", "L1", "L2s", "L2", "Lpls", "Lpl"]
TASKS = [
    # "abalone", "auction_verification", "bng_echoMonths", "california_housing", "infrared",
    # "life_expectancy", "ltfsid", "music_popularity", 
    "parkinsons_motor",
    "parkinsons_total", "swCSC"]

for task in TASKS:
    ddict = datasets.load_dataset("DaniFrame/AFRLA-instance-level-results", task, revision = "normalised")

    task_train = ddict["train"].to_pandas()
    task_val = ddict["validation"].to_pandas()
    task_test = ddict["test"].to_pandas()

    metrics = {
            err : {
                "train" : {
                    "RMSE" : None,
                    "MAPE" : None,
                    "R2" : None,
                    "Pearson" : None,
                    "Spearman" : None
                },
                "validation" : {
                    "RMSE" : None,
                    "MAPE" : None,
                    "R2" : None,
                    "Pearson" : None,
                    "Spearman" : None
                },
                "test" : {
                    "RMSE" : None,
                    "MAPE" : None,
                    "R2" : None,
                    "Pearson" : None,
                    "Spearman" : None
                }
            } for err in ERRORS
        }
    
    path = os.path.join("Experiments Data", task, "nn")

    results_train = task_train.copy()
    results_val = task_val.copy()
    results_test = task_test.copy()

    # I cannot be bothered to implement feature importance on neural networks
    for error in ERRORS:
        if error in {"Lpls", "Lpl"}:
            y_train, beta = compute_error(
                task_train["real"], task_train["prediction"], 
                error, beta = "mean", return_beta = True)

            y_val = compute_error(task_val["real"], task_val["prediction"], error, beta = beta)
            y_test = compute_error(task_test["real"], task_test["prediction"], error, beta = beta)

        else:
            y_train = compute_error(task_train["real"], task_train["prediction"], error)
            y_val = compute_error(task_val["real"], task_val["prediction"], error)
            y_test = compute_error(task_test["real"], task_test["prediction"], error)

            X_train = task_train.drop(columns = ["instance", "real", "prediction"])
            X_val = task_val.drop(columns = ["instance", "real", "prediction"])
            X_test = task_test.drop(columns = ["instance", "real", "prediction"])

        # Convert to dataloaders
        train_dl = neural.to_dataloader(X_train, y_train)
        val_dl = neural.to_dataloader(X_val, y_val, shuffle = False)
        test_dl = neural.to_dataloader(X_test, y_test, shuffle = False)

        # Create network
        layers = get_layers(X_train.shape[1])
        net = neural.create_network(
            input_size = X_train.shape[1],
            hidden_layers_sizes = layers,
            activations = ["relu"] * len(layers),
            output_size = 1)
        
        # Train network
        history, model = neural.train_network(net, train_dl, val_dl, epochs = 50)

        # Predictions
        with torch.no_grad():
            model.eval()
            y_train_pred = model(torch.tensor(X_train.values, dtype = torch.float32).to(device)).cpu().numpy().flatten()
            y_val_pred = model(torch.tensor(X_val.values, dtype = torch.float32).to(device)).cpu().numpy().flatten()
            y_test_pred = model(torch.tensor(X_test.values, dtype = torch.float32).to(device)).cpu().numpy().flatten()

        # Metrics
        metrics[error]["train"]["RMSE"] = root_mean_squared_error(y_train, y_train_pred)
        metrics[error]["train"]["MAPE"] = mean_absolute_percentage_error(y_train, y_train_pred)
        metrics[error]["train"]["R2"] = r2_score(y_train, y_train_pred)
        metrics[error]["train"]["Pearson"] = pearsonr(y_train, y_train_pred)[0]
        metrics[error]["train"]["Spearman"] = spearmanr(y_train, y_train_pred)[0]

        metrics[error]["validation"]["RMSE"] = root_mean_squared_error(y_val, y_val_pred)
        metrics[error]["validation"]["MAPE"] = mean_absolute_percentage_error(y_val, y_val_pred)
        metrics[error]["validation"]["R2"] = r2_score(y_val, y_val_pred)
        metrics[error]["validation"]["Pearson"] = pearsonr(y_val, y_val_pred)[0]
        metrics[error]["validation"]["Spearman"] = spearmanr(y_val, y_val_pred)[0]

        metrics[error]["test"]["RMSE"] = root_mean_squared_error(y_test, y_test_pred)
        metrics[error]["test"]["MAPE"] = mean_absolute_percentage_error(y_test, y_test_pred)
        metrics[error]["test"]["R2"] = r2_score(y_test, y_test_pred)
        metrics[error]["test"]["Pearson"] = pearsonr(y_test, y_test_pred)[0]
        metrics[error]["test"]["Spearman"] = spearmanr(y_test, y_test_pred)[0]

        # Add results onto results DataFrame
        results_train[f"{error}_real"] = y_train
        results_train[f"{error}_prediction"] = y_train_pred

        results_val[f"{error}_real"] = y_val
        results_val[f"{error}_prediction"] = y_val_pred

        results_test[f"{error}_real"] = y_test
        results_test[f"{error}_prediction"] = y_test_pred

        print("Task:", task, "Error:", error)

    # Save results (also shorten them by just saving index, real and prediction columns)
    results_train = results_train[["instance"] + [f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]]
    results_val = results_val[["instance"] + [f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]]
    results_test = results_test[["instance"] + [f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]]

    results_train.to_parquet(os.path.join(path, "train.parquet"))
    results_val.to_parquet(os.path.join(path, "validation.parquet"))
    results_test.to_parquet(os.path.join(path, "test.parquet"))

    # Save metrics
    with open(os.path.join(path, f"metrics.json"), "w") as f:
        json.dump(metrics, f, indent = 4)

    # Save extra params
    if error in {"Lpls", "Lpl"}:
        with open(os.path.join("Experiments Data", task, "nn", f"extra_params.json"), "w") as f:
            json.dump({"Lpls_beta" : beta}, f, indent = 4)

    print("-" * 45)
    print("Task:", task, "Completed")
    print("-" * 45)