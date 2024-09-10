import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_score_heatmap(df, model, remove = ["ltfsid"], metric = "spearman_significance"):
    df = df[df["model"] == model]

    # Remove rows where value is in remove list
    df = df[~df["task"].isin(remove)]

    # Groupby
    gdf = df.groupby(["proxy", "target"])[metric].sum().reset_index()
    matrix = gdf.pivot(columns = ["proxy"], index = ["target"], values = metric)

    ax = sns.heatmap(
        data = matrix, annot = True, center = 0, square = True,
        cmap =  sns.diverging_palette(10, 150, s = 100, l = 50, n = 100, center = "light"), 
        vmax = 10, vmin = -10, cbar_kws = {"ticks": np.arange(-10, 11, 2)},
        linewidths = 0.4, linecolor = "lightgrey")

    ax.invert_yaxis()

    for _, spine in ax.spines.items(): 
        spine.set_visible(True) 
        spine.set_linewidth(0.5)
        spine.set_color("black")

    return ax

def get_diff_heatmap(df, model, remove = ["ltfsid"], metric = "spearman", significant_only = False):
    df = df[df["model"] == model]

    # Remove rows where value is in remove list
    df = df[~df["task"].isin(remove)]

    # Groupby
    if significant_only:
        gdf = df.loc[df[f"{metric}_significance"] != 0].groupby(["proxy", "target"])[f"{metric}_diff"].mean().reset_index()
    else:
        gdf = df.groupby(["proxy", "target"])[f"{metric}_diff"].mean().reset_index()

    matrix = gdf.pivot(columns = ["proxy"], index = ["target"], values = f"{metric}_diff")

    # Get max and min for heatmap. Obtain the closes multiple of 0.05 to the max and min values
    max_val = np.ceil(matrix.max().max() / 0.05) * 0.05
    min_val = np.floor(matrix.min().min() / 0.05) * 0.05

    limit = max(max_val, np.abs(min_val))

    ax = sns.heatmap(
        data = matrix, annot = True, center = 0, fmt = ".3f", square = True,
        cmap =  sns.diverging_palette(10, 150, s = 100, l = 50, n = 100, center = "light"), 
        vmax = limit, vmin = -limit, cbar_kws = {"ticks": np.arange(-limit, limit + 0.01, 0.05)},
        linewidths = 0.4, linecolor = "lightgrey")

    ax.invert_yaxis()

    for _, spine in ax.spines.items(): 
        spine.set_visible(True) 
        spine.set_linewidth(0.5)
        spine.set_color("black")

    return ax

if __name__ == "__main__":
    df = pd.read_csv("Comparisons Data/significance.csv")

    MODELS = ["xgboost", "lr", "nn", "bayreg"]

    for model in MODELS:
        score_heatmap = get_score_heatmap(df, model)
        score_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/spearman_score_heatmap.png")
        plt.close()

        diff_heatmap = get_diff_heatmap(df, model)
        diff_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/spearman_diff_heatmap.png")
        plt.close()

        sign_diff_heatmap = get_diff_heatmap(df, model, significant_only = True)
        sign_diff_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/spearman_diff_significant_heatmap.png")
        plt.close()

        # score_heatmap = get_score_heatmap(df, model, metric = "pearson_significance")
        # score_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/pearson_score_heatmap.png")
        # plt.close()

        # diff_heatmap = get_diff_heatmap(df, model, metric = "pearson")
        # diff_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/pearson_diff_heatmap.png")
        # plt.close()

        # sign_diff_heatmap = get_diff_heatmap(df, model, metric = "pearson", significant_only = True)
        # sign_diff_heatmap.get_figure().savefig(f"Plots/Heatmaps/{model}/pearson_diff_significant_heatmap.png")
        # plt.close()