from sklearn.metrics import root_mean_squared_error, r2_score
from pyassessors.transformations import transform_error
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import resample

from tqdm import tqdm

import pandas as pd
import json
import os

PAIRS = [
    ("L1s", "L1"), ("L1s", "L2s"), ("L1s", "L2"), ("L1s", "Lpls"), ("L1s", "Lpl"),
    ("L2s", "L1s"), ("L2s", "L1"), ("L2s", "L2"), ("L2s", "Lpls"), ("L2s", "Lpl"),
    ("Lpls", "L1s"), ("Lpls", "L1"), ("Lpls", "L2s"), ("Lpls", "L2"), ("Lpls", "Lpl"),
    ("L1", "L2"), ("L1", "Lpl"),
    ("L2", "L1"), ("L2", "Lpl"),
    ("Lpl", "L1"), ("Lpl", "L2")
]

def get_beta(base_path, task, model):
    path = os.path.join(base_path, task, model, "extra_params.json")

    with open(path, "r") as f:
        params = json.load(f)

    return params["Lpls_beta"]

def compare_proxy_target(
        task, model, proxy, target, base_path, 
        seed = None, n_bootstraps = 1000):
    
    if seed is None:
        path = os.path.join(base_path, task, model, "test.parquet")
    else:
        path = os.path.join(base_path, task, model, f"seed_{seed}", "test.parquet")
    
    df = pd.read_parquet(path)

    proxy_pred = df[f"{proxy}_prediction"].values
    target_pred = df[f"{target}_prediction"].values
    target_real = df[f"{target}_real"].values

    beta = get_beta(base_path, task, model)

    # Transform proxy error
    if proxy not in {"Lpls", "Lpl"}:
        if target in {"Lpls", "Lpl"}:
            proxy_pred_trans = transform_error(
                error = proxy_pred,
                from_error = proxy,
                to_error = target,
                beta = beta
            )
        else:
            proxy_pred_trans = transform_error(
                error = proxy_pred,
                from_error = proxy,
                to_error = target
            )
    else:
        if target == "Lpl":
            proxy_pred_trans = transform_error(
                error = proxy_pred,
                from_error = proxy,
                to_error = target)
        else:
            proxy_pred_trans = transform_error(
                error = proxy_pred,
                from_error = proxy,
                to_error = target,
                beta = beta)

    base_df = pd.DataFrame({
        "proxy_trans_pred" : proxy_pred_trans,
        "target_pred": target_pred,
        "target_real": target_real,
    })

    proxy_rmses = []
    proxy_r2s = []
    proxy_pearsons = []
    proxy_spearmans = []

    target_rmses = []
    target_r2s = []
    target_pearsons = []
    target_spearmans = []

    # Get metrics via bootstrapping
    for _ in tqdm(range(n_bootstraps), desc = "Bootstrapping..."):
        boot_df = resample(
            base_df, replace = True, n_samples = len(base_df))
        
        boot_proxy = boot_df["proxy_trans_pred"]
        boot_target = boot_df["target_pred"]
        boot_actual = boot_df["target_real"]

        proxy_rmse = root_mean_squared_error(boot_proxy, boot_actual)
        proxy_r2 = r2_score(boot_proxy, boot_actual)
        proxy_pearson = pearsonr(boot_proxy, boot_actual)[0]
        proxy_spearman = spearmanr(boot_proxy, boot_actual)[0]

        target_rmse = root_mean_squared_error(boot_target, boot_actual)
        target_r2 = r2_score(boot_target, boot_actual)
        target_pearson = pearsonr(boot_target, boot_actual)[0]
        target_spearman = spearmanr(boot_target, boot_actual)[0]

        proxy_rmses.append(proxy_rmse)
        proxy_r2s.append(proxy_r2)
        proxy_pearsons.append(proxy_pearson)
        proxy_spearmans.append(proxy_spearman)

        target_rmses.append(target_rmse)
        target_r2s.append(target_r2)
        target_pearsons.append(target_pearson)
        target_spearmans.append(target_spearman)

    return pd.DataFrame({
        "proxy_rmse": proxy_rmses,
        "proxy_r2": proxy_r2s,
        "proxy_pearson": proxy_pearsons,
        "proxy_spearman": proxy_spearmans,
        "target_rmse": target_rmses,
        "target_r2": target_r2s,
        "target_pearson": target_pearsons,
        "target_spearman": target_spearmans
    })

if __name__ == "__main__":
    base_path = "Experiments Data"
    end_path = "Comparisons Data"

    TASKS = [
        "abalone", 
        "auction_verification", "bng_echoMonths", "california_housing", 
        "infrared", "life_expectancy", "ltfsid", "music_popularity", 
        "parkinsons_motor", "parkinsons_total", "swCSC"]
    MODELS = ["bayreg"]
    SEEDS = [13, 290, 420, 478, 504]

    for task in TASKS:
        for model in MODELS:
            if model == "xgboost":
                for proxy, target in PAIRS:
                    boot_dfs = []
                    for seed in SEEDS:
                        print(f"\nComparing {proxy} and {target} for {task} using {model} (seed {seed})...")
                        boot_df = compare_proxy_target(task, model, proxy, target, base_path, seed = seed, n_bootstraps = 200)
                        boot_dfs.append(boot_df)

                    pd.concat(boot_dfs).to_csv(
                        os.path.join(end_path, task, model, f"{proxy}_{target}.csv"), index = False)

            else:
                for proxy, target in PAIRS:
                    print(f"\nComparing {proxy} and {target} for {task} using {model}...")
                    boot_df = compare_proxy_target(task, model, proxy, target, base_path, n_bootstraps = 1000)
                    boot_df.to_csv(os.path.join(end_path, task, model, f"{proxy}_{target}.csv"), index = False)