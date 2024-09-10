from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from pyassessors.errors import compute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import datasets
import json
import os

ERRORS = ["L1s", "L1", "L2s", "L2", "Lpls", "Lpl"]
TASKS = [
    "abalone", "auction_verification", "bng_echoMonths", "california_housing", "infrared",
    "life_expectancy", "ltfsid", "music_popularity", "parkinsons_motor",
    "parkinsons_total", "swCSC"]

for task in TASKS:
    ddict = datasets.load_dataset("DaniFrame/AFRLA-instance-level-results", task)

    task_train = ddict["train"].to_pandas()
    task_test = pd.concat([ddict["validation"].to_pandas(), ddict["test"].to_pandas()], ignore_index = True)

    task_train = pd.get_dummies(task_train, dtype = np.int32)
    task_test = pd.get_dummies(task_test, dtype = np.int32)

    metrics = {
        err : {
            "train" : {
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

    path = os.path.join("Experiments Data", task, "lr")

    results_train = task_train.copy()
    results_test = task_test.copy()

    feature_importance_df = pd.DataFrame()

    for error in ERRORS:

        if error in {"Lpls", "Lpl"}:
            y_train, beta = compute_error(
                task_train["real"], task_train["prediction"], 
                error, beta = "mean", return_beta = True)

            y_test = compute_error(task_test["real"], task_test["prediction"], error, beta = beta)

        else:
            y_train = compute_error(task_train["real"], task_train["prediction"], error)
            y_test = compute_error(task_test["real"], task_test["prediction"], error)

            X_train = task_train.drop(columns = ["instance", "real", "prediction"])
            X_test = task_test.drop(columns = ["instance", "real", "prediction"])

        # Train model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Feature importance
        feature_importance_df.index = lr.feature_names_in_
        feature_importance_df[f"{error}_importance"] = lr.coef_

        # Predictions
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)

        # Metrics
        metrics[error]["train"]["RMSE"] = root_mean_squared_error(y_train, y_train_pred)
        metrics[error]["train"]["MAPE"] = mean_absolute_percentage_error(y_train, y_train_pred)
        metrics[error]["train"]["R2"] = r2_score(y_train, y_train_pred)
        metrics[error]["train"]["Pearson"] = pearsonr(y_train, y_train_pred)[0]
        metrics[error]["train"]["Spearman"] = spearmanr(y_train, y_train_pred)[0]

        metrics[error]["test"]["RMSE"] = root_mean_squared_error(y_test, y_test_pred)
        metrics[error]["test"]["MAPE"] = mean_absolute_percentage_error(y_test, y_test_pred)
        metrics[error]["test"]["R2"] = r2_score(y_test, y_test_pred)
        metrics[error]["test"]["Pearson"] = pearsonr(y_test, y_test_pred)[0]
        metrics[error]["test"]["Spearman"] = spearmanr(y_test, y_test_pred)[0]

        # Add results onto results DataFrame
        results_train[f"{error}_real"] = y_train
        results_train[f"{error}_prediction"] = y_train_pred

        results_test[f"{error}_real"] = y_test
        results_test[f"{error}_prediction"] = y_test_pred

        print("Task:", task, "Error:", error)

    # Save feature importance
    feature_importance_df.to_csv(os.path.join(path, "feature_importance.csv"))

    # Save results (also shorten them by just saving index, real and prediction columns)
    results_train = results_train[[f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]].reset_index()
    results_test = results_test[[f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]].reset_index()

    results_train.to_parquet(os.path.join(path, "train.parquet"))
    results_test.to_parquet(os.path.join(path, "test.parquet"))

    # Save metrics
    with open(os.path.join(path, f"metrics.json"), "w") as f:
        json.dump(metrics, f, indent = 4)

    # Save extra params
    if error in {"Lpls", "Lpl"}:
        with open(os.path.join("Experiments Data", task, "lr", f"extra_params.json"), "w") as f:
            json.dump({"Lpls_beta" : beta}, f, indent = 4)

    print("-" * 45)
    print("Task:", task, "Completed")
    print("-" * 45)