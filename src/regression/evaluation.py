import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = pearsonr(y_true, y_pred).statistic

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def top_k_mean_affinity(y_selected, k=20):
    k = min(k, len(y_selected))
    return np.mean(np.sort(y_selected)[-k:])


def best_affinity(y_selected):
    return np.max(y_selected)


def top_k_enrichment(y_selected, y_all, k=100):
    global_top_k_threshold = np.sort(y_all)[-k]

    n_selected_top_k = np.sum(y_selected >= global_top_k_threshold)

    return n_selected_top_k / k