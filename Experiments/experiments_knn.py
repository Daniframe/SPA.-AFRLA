from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from pyassessors.errors import compute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import datasets
import xgboost
import json
import os

raise NotImplementedError("This script is not yet implemented.")

SEEDS = [13, 290, 420, 478, 504]
ERRORS = ["L1s", "L1", "L2s", "L2", "Lpls", "Lpl"]
TASKS = [
    "abalone", "auction_verification", "bng_echoMonths", "california_housing", "infrared",
    "life_expectancy", "ltfsid", "music_popularity", "parkinsons_motor",
    "parkinsons_total", "swCSC"]

for task in TASKS:
    ddict = datasets.load_dataset("DaniFrame/AFRLA-instance-level-results", task)

    task_train = ddict["train"].to_pandas()
    task_val = ddict["validation"].to_pandas()
    task_test = ddict["test"].to_pandas()

    task_train = pd.get_dummies(task_train, dtype = np.int32)
    task_val = pd.get_dummies(task_val, dtype = np.int32)
    task_test = pd.get_dummies(task_test, dtype = np.int32)

    for seed in SEEDS:

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

        path = os.path.join("Experiments Data", task, "xgboost", f"seed_{seed}")

        results_train = task_train.copy()
        results_val = task_val.copy()
        results_test = task_test.copy()

        feature_importance_df = pd.DataFrame()

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

            xgb_reg = xgboost.XGBRegressor(
                objective = "reg:squarederror", 
                n_estimators = 1000, 
                max_depth = 7, 
                colsample_bytree = 0.8,
                subsample = 0.8,
                learning_rate = 0.05, 
                n_jobs = -1,
                early_stopping_rounds = 25,
                random_state = seed)
            
            xgb_reg.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False)

            # Feature importance
            feature_importance_df.index = xgb_reg.feature_names_in_
            feature_importance_df[f"{error}_importance"] = xgb_reg.feature_importances_

            # Predictions
            y_train_pred = xgb_reg.predict(X_train)
            y_val_pred = xgb_reg.predict(X_val)
            y_test_pred = xgb_reg.predict(X_test)

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

            print("Task:", task, "Seed:", seed, "Error:", error)

        # Save feature importance
        feature_importance_df.to_csv(os.path.join(path, "feature_importance.csv"))

        # Save results (also shorten them by just saving index, real and prediction columns)
        results_train = results_train[[f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]].reset_index()
        results_val = results_val[[f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]].reset_index()
        results_test = results_test[[f"{err}_real" for err in ERRORS] + [f"{err}_prediction" for err in ERRORS]].reset_index()

        results_train.to_parquet(os.path.join(path, "train.parquet"))
        results_val.to_parquet(os.path.join(path, "validation.parquet"))
        results_test.to_parquet(os.path.join(path, "test.parquet"))

        # Save metrics
        with open(os.path.join(path, f"metrics.json"), "w") as f:
            json.dump(metrics, f, indent = 4)

    # Save extra params
    if error in {"Lpls", "Lpl"}:
        with open(os.path.join("Experiments Data", task, "xgboost", f"extra_params.json"), "w") as f:
            json.dump({"Lpls_beta" : beta}, f, indent = 4)

    print("-" * 45)
    print("Task:", task, "Completed")
    print("-" * 45)