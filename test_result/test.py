import json
import re
import argparse
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from scipy.stats import pearsonr, spearmanr
import math

def extract_number(text):
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else None

def is_binary(values):
    return set(values).issubset({0, 1})

def compute_metrics(preds, targets, selected_metrics):
    results = {}

    if is_binary(preds) and is_binary(targets):
        print("Task type: Binary Classification")
        if "auc" in selected_metrics:
            results["AUC-ROC"] = roc_auc_score(targets, preds)
        if "pr_auc" in selected_metrics:
            results["PR-AUC"] = average_precision_score(targets, preds)
        if "acc" in selected_metrics:
            results["Accuracy"] = accuracy_score(targets, preds)
        if "precision" in selected_metrics:
            results["Precision"] = precision_score(targets, preds)
        if "recall" in selected_metrics:
            results["Recall"] = recall_score(targets, preds)
        if "f1" in selected_metrics:
            results["F1 Score"] = f1_score(targets, preds)
    else:
        print("Task type: Regression")
        if "mse" in selected_metrics:
            results["MSE"] = mean_squared_error(targets, preds)
        if "rmse" in selected_metrics:
            results["RMSE"] = math.sqrt(mean_squared_error(targets, preds))
        if "mae" in selected_metrics:
            results["MAE"] = mean_absolute_error(targets, preds)
        if "r2" in selected_metrics:
            results["R^2"] = r2_score(targets, preds)
        if "pcc" in selected_metrics:
            results["PCC"] = pearsonr(targets, preds)[0]
        if "scc" in selected_metrics:
            results["SCC"] = spearmanr(targets, preds)[0]

    return results

def load_predictions(filename):
    preds = []
    targets = []
    with open(filename, "r") as f:
        for line in f:
            item = json.loads(line)
            pred = extract_number(item["prediction"])
            target = extract_number(item["target"])
            if pred is not None and target is not None:
                preds.append(pred)
                targets.append(target)
    return np.array(preds), np.array(targets)

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions with selected metrics.")
    parser.add_argument("--file", type=str, required=True, help="Path to predictions.txt")
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        help="List of metrics to compute. Supported: mse, rmse, mae, r2, pcc, scc, auc, pr_auc, acc, precision, recall, f1"
    )
    args = parser.parse_args()

    predictions, targets = load_predictions(args.file)
    metrics = compute_metrics(predictions, targets, args.metrics)

    print("\nEvaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
