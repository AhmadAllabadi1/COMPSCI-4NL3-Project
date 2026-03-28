"""
Shared utilities for baseline models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent       # final_step/baselines/
DATA_DIR = BASE_DIR.parent                       # final_step/
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

LABEL_LIST = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].str.strip().str.upper()
    return df


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_json(path: Path, data: dict) -> None:
    import json
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_LIST)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0)
    save_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {save_path}")
    return report


def save_misclassified(texts, y_true, y_pred, save_path, max_samples=50):
    rows = []
    for text, t, p in zip(texts, y_true, y_pred):
        if t != p:
            rows.append({"text": text, "true_label": t, "predicted_label": p})
    df = pd.DataFrame(rows[:max_samples])
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path} ({len(df)} examples)")


def evaluate_and_save(name, y_true, y_pred, texts, split_name):
    """Evaluate predictions and save all outputs. Returns metrics dict."""
    prefix = f"{name}_{split_name}"
    metrics = compute_metrics(y_true, y_pred)

    save_json(RESULTS_DIR / f"{prefix}_metrics.json", metrics)
    save_classification_report(y_true, y_pred,
                               RESULTS_DIR / f"{prefix}_classification_report.txt")
    plot_confusion_matrix(y_true, y_pred,
                          RESULTS_DIR / f"{prefix}_confusion_matrix.png",
                          f"{name.replace('_', ' ').title()} — {split_name.title()} Set")
    save_misclassified(texts, y_true, y_pred,
                       RESULTS_DIR / f"{prefix}_misclassified.csv")

    return metrics
