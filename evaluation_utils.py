"""
evaluation_utils.py

All the evaluation and output-saving logic lives here:
  - evaluate_model(): computes metrics, saves classification report,
    confusion matrix, and error analysis CSV for a single model
  - run_comparative_analysis(): compares baseline vs DistilBERT predictions
    side by side and saves the result
  - ensure_output_dir(): just makes sure the outputs folder exists
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of showing a window
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# all outputs go into this folder
OUTPUT_DIR = Path(__file__).parent / "outputs"


def ensure_output_dir():
    """Create the outputs/ folder if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"[INFO] Outputs will be saved to: {OUTPUT_DIR.resolve()}")


def evaluate_model(model_name, test_texts, test_labels, predictions, label_names):
    """
    Evaluates a model's predictions and saves everything we need for the paper.

    What gets saved:
      - a classification report as CSV (per-class precision, recall, f1)
      - a confusion matrix as PNG (heatmap)
      - an error analysis CSV (every misclassified example with metadata)

    Args:
        model_name:   string like "baseline" or "distilbert" (used in filenames)
        test_texts:   the original test strings
        test_labels:  true labels as integers
        predictions:  predicted labels as integers
        label_names:  list of human-readable class names

    Returns:
        metrics: dict with accuracy, macro-f1, macro-precision, macro-recall
    """
    print(f"Evaluation: {model_name.upper()}")

    # 1. print the full classification report to console
    report_str = classification_report(
        test_labels, predictions,
        target_names=label_names,
        zero_division=0,
    )
    print(report_str)

    # also save it as a CSV since that's easier to use in the paper
    report_dict = classification_report(
        test_labels, predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = OUTPUT_DIR / f"{model_name}_classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"Saved Classification report -> {report_csv_path}")

    # 2. compute the overall numbers
    acc = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro", zero_division=0
    )
    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro-F1": round(f1, 4),
        "Macro-Precision": round(precision, 4),
        "Macro-Recall": round(recall, 4),
    }
    print(f"Accuracy:\t\t{acc:.4f}")
    print(f"Macro-F1:\t\t{f1:.4f}")
    print(f"Macro-Precision:\t\t{precision:.4f}")
    print(f"Macro-Recall:\t\t{recall:.4f}")

    # 3. plot and save the confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name.upper()}", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved Confusion matrix -> {cm_path}")

    # 4. save every wrong prediction to a CSV for error analysis
    errors = []
    for i in range(len(test_labels)):
        if predictions[i] != test_labels[i]:
            errors.append({
                "text": test_texts[i],
                "true_label": label_names[test_labels[i]],
                "predicted_label": label_names[predictions[i]],
                "text_length": len(test_texts[i]),
            })

    errors_df = pd.DataFrame(errors)
    errors_csv_path = OUTPUT_DIR / f"{model_name}_errors.csv"
    errors_df.to_csv(errors_csv_path, index=False)
    print(f"Error analysis ({len(errors)} misclassified) -> {errors_csv_path}")

    return metrics


def run_comparative_analysis(
    test_texts, test_labels, baseline_preds, distilbert_preds, label_names
):
    """
    Puts both models' predictions side by side so you can compare them.
    
    This is super useful for the ablation study section of the paper.
    You can filter the CSV to find examples where:
      - only the baseline got it right
      - only DistilBERT got it right
      - both failed (probably ambiguous examples)
    
    Saves to: outputs/comparative_error_analysis.csv
    """
    print(f"Comparative Analysis:")

    rows = []
    for i in range(len(test_labels)):
        rows.append({
            "text": test_texts[i],
            "true_label": label_names[test_labels[i]],
            "baseline_pred": label_names[baseline_preds[i]],
            "distilbert_pred": label_names[distilbert_preds[i]],
            "baseline_correct": bool(baseline_preds[i] == test_labels[i]),
            "distilbert_correct": bool(distilbert_preds[i] == test_labels[i]),
        })

    comp_df = pd.DataFrame(rows)

    comp_path = OUTPUT_DIR / "comparative_error_analysis.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved Comparative analysis -> {comp_path}")

    # quick breakdown of how they compared
    both_correct    = comp_df["baseline_correct"] & comp_df["distilbert_correct"]
    both_wrong      = ~comp_df["baseline_correct"] & ~comp_df["distilbert_correct"]
    only_baseline   = comp_df["baseline_correct"] & ~comp_df["distilbert_correct"]
    only_distilbert = ~comp_df["baseline_correct"] & comp_df["distilbert_correct"]

    print(f"Both correct:\t\t{both_correct.sum():4d} samples")
    print(f"Both wrong:\t\t{both_wrong.sum():4d} samples")
    print(f"Only baseline correct:\t{only_baseline.sum():4d} samples")
    print(f"Only DistilBERT correct:\t{only_distilbert.sum():4d} samples")
    print(f"Total test samples:\t{len(test_labels):4d} samples")
