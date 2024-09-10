import pandas as pd
import numpy as np
import os



def is_overlap(proxy_conf, target_conf):
    # Check if the two confidence intervals overlap
    return proxy_conf[1] >= target_conf[0] and proxy_conf[0] <= target_conf[1]

def is_higher_interval(proxy_conf, target_conf):
    # Check if the proxy confidence interval is higher than the target confidence interval
    return proxy_conf[0] > target_conf[1]

def obtain_significance(task, model, proxy, target, base_path):
    path = os.path.join(base_path, task, model,f"{proxy}_{target}.csv")
    df = pd.read_csv(path)

    rmse_proxy_conf = np.percentile(df["proxy_rmse"], [2.5, 97.5])
    r2_proxy_conf = np.percentile(df["proxy_r2"], [2.5, 97.5])
    pearson_proxy_conf = np.percentile(df["proxy_pearson"], [2.5, 97.5])
    spearman_proxy_conf = np.percentile(df["proxy_spearman"], [2.5, 97.5])

    rmse_target_conf = np.percentile(df["target_rmse"], [2.5, 97.5])
    r2_target_conf = np.percentile(df["target_r2"], [2.5, 97.5])
    pearson_target_conf = np.percentile(df["target_pearson"], [2.5, 97.5])
    spearman_target_conf = np.percentile(df["target_spearman"], [2.5, 97.5])

    rmse_significance = not is_overlap(rmse_proxy_conf, rmse_target_conf)
    r2_significance = not is_overlap(r2_proxy_conf, r2_target_conf)
    pearson_significance = not is_overlap(pearson_proxy_conf, pearson_target_conf)
    spearman_significance = not is_overlap(spearman_proxy_conf, spearman_target_conf)

    if rmse_significance:
        if is_higher_interval(rmse_proxy_conf, rmse_target_conf):
            rmse_significance = 1
        else:
            rmse_significance = -1
    else:
        rmse_significance = 0

    if r2_significance:
        if is_higher_interval(r2_proxy_conf, r2_target_conf):
            r2_significance = 1
        else:
            r2_significance = -1
    else:
        r2_significance = 0

    if pearson_significance:
        if is_higher_interval(pearson_proxy_conf, pearson_target_conf):
            pearson_significance = 1
        else:
            pearson_significance = -1
    else:
        pearson_significance = 0
    
    if spearman_significance:
        if is_higher_interval(spearman_proxy_conf, spearman_target_conf):
            spearman_significance = 1
        else:
            spearman_significance = -1
    else:
        spearman_significance = 0

    return [
        task, model, proxy, target, 
        rmse_significance, r2_significance, pearson_significance, spearman_significance,
        df["proxy_rmse"].mean() - df["target_rmse"].mean(),
        df["proxy_r2"].mean() - df["target_r2"].mean(),
        df["proxy_pearson"].mean() - df["target_pearson"].mean(),
        df["proxy_spearman"].mean() - df["target_spearman"].mean()]

PAIRS = [
    ("L1s", "L1"), ("L1s", "L2s"), ("L1s", "L2"), ("L1s", "Lpls"), ("L1s", "Lpl"),
    ("L2s", "L1s"), ("L2s", "L1"), ("L2s", "L2"), ("L2s", "Lpls"), ("L2s", "Lpl"),
    ("Lpls", "L1s"), ("Lpls", "L1"), ("Lpls", "L2s"), ("Lpls", "L2"), ("Lpls", "Lpl"),
    ("L1", "L2"), ("L1", "Lpl"),
    ("L2", "L1"), ("L2", "Lpl"),
    ("Lpl", "L1"), ("Lpl", "L2")
]

TASKS = [
    "abalone", "auction_verification", "bng_echoMonths", "california_housing", 
    "infrared", "life_expectancy", "ltfsid", "music_popularity", 
    "parkinsons_motor", "parkinsons_total", "swCSC"]
MODELS = ["lr", "xgboost", "nn", "bayreg"]

if __name__ == "__main__":
    base_path = "Comparisons Data"
    end_path = "Comparisons Data"

    results = []
    
    for task in TASKS:
        for model in MODELS:
            for proxy, target in PAIRS:
                print(f"\nObtaining significance for {proxy} and {target} for {task} using {model}...")
                results.append(obtain_significance(task, model, proxy, target, base_path))

    results_df = pd.DataFrame(results, columns = [
        "task", "model", "proxy", "target", 
        "rmse_significance", "r2_significance", "pearson_significance", "spearman_significance",
        "rmse_diff", "r2_diff", "pearson_diff", "spearman_diff"])
    results_df.to_csv(os.path.join(end_path, "significance.csv"), index = False)