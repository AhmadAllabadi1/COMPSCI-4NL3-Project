"""
DistilBERT Fine-Tuning for Reddit Advice Intent Classification
===============================================================
- Loads train/val/test CSVs from ../train.csv, ../validation.csv, ../test.csv
- Optionally applies EDA data augmentation on the training set
- Fine-tunes distilbert-base-uncased with HuggingFace Trainer API
- Runs ablation: trains once WITHOUT augmentation, once WITH augmentation
- Produces all evaluation outputs and visualizations in ./diagrams/

Usage:
    python train_distilbert.py                    # runs full ablation (no-aug + aug)
    python train_distilbert.py --no-augmentation  # only train without augmentation
    python train_distilbert.py --augmentation     # only train with augmentation
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from eda_augment import augment_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # final_step/ahmad_model/
DATA_DIR = BASE_DIR.parent                          # final_step/
DIAGRAM_DIR = BASE_DIR / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
SEED = 42

# Label mapping (alphabetical for consistency)
LABEL_LIST = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path: Path, has_labels: bool = True) -> pd.DataFrame:
    """Load CSV and normalize label column if present."""
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).fillna("")
    if has_labels:
        df["label"] = df["label"].str.strip().str.upper()
    return df


def df_to_dataset(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> Dataset:
    """Convert a DataFrame to a HuggingFace Dataset with tokenized inputs."""
    data_dict = {"text": df["text"].tolist()}
    if has_labels:
        data_dict["label"] = [LABEL2ID[l] for l in df["label"]]
    dataset = Dataset.from_dict(data_dict)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return dataset


# ---------------------------------------------------------------------------
# Metrics for Trainer
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute metrics used during training evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_label_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame, save_dir: Path) -> None:
    """Bar charts showing label distribution for train/val/test."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
        if "label" in df.columns:
            counts = df["label"].value_counts().reindex(LABEL_LIST, fill_value=0)
            colors = sns.color_palette("viridis", len(LABEL_LIST))
            bars = ax.bar(LABEL_LIST, counts.values, color=colors)
            ax.bar_label(bars, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No labels\n(test set)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
        ax.set_title(f"{name} Set (n={len(df)})", fontsize=13)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'label_distribution.png'}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_path: Path,
                          title: str = "Confusion Matrix") -> None:
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_history: list, save_path: Path,
                         title: str = "Training & Validation Loss") -> None:
    """Plot training and validation loss curves from Trainer log history."""
    train_loss = [(entry["epoch"], entry["loss"])
                  for entry in log_history if "loss" in entry and "eval_loss" not in entry]
    eval_loss = [(entry["epoch"], entry["eval_loss"])
                 for entry in log_history if "eval_loss" in entry]

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        epochs_t, losses_t = zip(*train_loss)
        ax.plot(epochs_t, losses_t, "o-", label="Train Loss", markersize=4)
    if eval_loss:
        epochs_e, losses_e = zip(*eval_loss)
        ax.plot(epochs_e, losses_e, "s-", label="Validation Loss", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_classification_report(y_true, y_pred, label_names, save_path: Path) -> str:
    """Save classification report as text and return the string."""
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    save_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {save_path}")
    return report


def save_misclassified(texts, y_true, y_pred, label_names, save_path: Path,
                       max_samples: int = 50) -> None:
    """Export misclassified examples to CSV for error analysis."""
    misclassified = []
    for text, true_id, pred_id in zip(texts, y_true, y_pred):
        if true_id != pred_id:
            misclassified.append({
                "text": text,
                "true_label": label_names[true_id],
                "predicted_label": label_names[pred_id],
            })
    df = pd.DataFrame(misclassified[:max_samples])
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path} ({len(df)} examples)")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train_and_evaluate(
    train_dataset: Dataset,
    val_dataset: Dataset,
    val_texts: list,
    run_name: str,
    output_dir: Path,
    diagram_dir: Path,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
) -> dict:
    """
    Fine-tune DistilBERT and evaluate on the validation set.

    Returns a dict with metrics and the trained model/trainer.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"{'='*60}\n")

    # Load a fresh model for each run
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=0.3,  # dropout on classification head
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_dir=str(output_dir / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n  Eval results for {run_name}:")
    for k, v in eval_results.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Get predictions on validation set
    predictions = trainer.predict(val_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = np.array(val_dataset["label"])

    # --- Save all outputs ---
    prefix = run_name.lower().replace(" ", "_")

    # Confusion matrix
    plot_confusion_matrix(
        true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"{prefix}_confusion_matrix.png",
        title=f"Confusion Matrix — {run_name}",
    )

    # Classification report
    report_str = save_classification_report(
        true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"{prefix}_classification_report.txt",
    )
    print(f"\n  Classification Report ({run_name}):\n{report_str}")

    # Training/validation loss curves
    plot_training_curves(
        trainer.state.log_history,
        diagram_dir / f"{prefix}_loss_curves.png",
        title=f"Loss Curves — {run_name}",
    )

    # Misclassified examples
    save_misclassified(
        val_texts, true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"{prefix}_misclassified.csv",
    )

    # Metrics JSON
    metrics = {
        "accuracy": accuracy_score(true_ids, pred_ids),
        "f1_macro": f1_score(true_ids, pred_ids, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "precision_macro": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "recall_macro": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
    }
    metrics_path = diagram_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved: {metrics_path}")

    # Save the best model
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    print(f"  Best model saved to: {best_model_dir}")

    return {
        "metrics": metrics,
        "trainer": trainer,
        "predictions": pred_ids,
        "true_labels": true_ids,
    }


# ---------------------------------------------------------------------------
# Ablation comparison chart
# ---------------------------------------------------------------------------
def plot_ablation_comparison(metrics_no_aug: dict, metrics_aug: dict,
                             save_path: Path) -> None:
    """Side-by-side bar chart comparing no-aug vs aug metrics."""
    metric_names = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    no_aug_vals = [metrics_no_aug[m] for m in metric_names]
    aug_vals = [metrics_aug[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, no_aug_vals, width, label="Without Augmentation",
                   color="#4C72B0")
    bars2 = ax.bar(x + width / 2, aug_vals, width, label="With EDA Augmentation",
                   color="#DD8452")

    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: With vs Without EDA Augmentation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DistilBERT fine-tuning for advice intent classification")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Only run without augmentation")
    parser.add_argument("--augmentation", action="store_true",
                        help="Only run with augmentation")
    parser.add_argument("--num-aug", type=int, default=4,
                        help="Number of augmented samples per original (default: 4)")
    parser.add_argument("--eda-alpha", type=float, default=0.1,
                        help="EDA intensity parameter (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    args = parser.parse_args()

    set_seed(SEED)

    # Determine which runs to do
    run_no_aug = not args.augmentation  # run unless --augmentation only
    run_aug = not args.no_augmentation  # run unless --no-augmentation only

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    train_df = load_data(TRAIN_CSV, has_labels=True)
    val_df = load_data(VAL_CSV, has_labels=True)
    test_df = load_data(TEST_CSV, has_labels=False)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # Label distribution plot (always produced)
    # ------------------------------------------------------------------
    print("\nPlotting label distributions...")
    plot_label_distribution(train_df, val_df, test_df, DIAGRAM_DIR)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize validation set (shared across runs)
    val_dataset = df_to_dataset(val_df, tokenizer, has_labels=True)
    val_texts = val_df["text"].tolist()

    results = {}

    # ------------------------------------------------------------------
    # Run 1: Without augmentation
    # ------------------------------------------------------------------
    if run_no_aug:
        train_dataset_no_aug = df_to_dataset(train_df, tokenizer, has_labels=True)
        results["no_aug"] = train_and_evaluate(
            train_dataset=train_dataset_no_aug,
            val_dataset=val_dataset,
            val_texts=val_texts,
            run_name="No Augmentation",
            output_dir=BASE_DIR / "run_no_aug",
            diagram_dir=DIAGRAM_DIR,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Run 2: With EDA augmentation
    # ------------------------------------------------------------------
    if run_aug:
        print(f"\nApplying EDA augmentation (num_aug={args.num_aug}, alpha={args.eda_alpha})...")
        aug_texts, aug_labels = augment_dataset(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            num_aug=args.num_aug,
            alpha=args.eda_alpha,
        )
        # Combine original + augmented
        combined_texts = train_df["text"].tolist() + aug_texts
        combined_labels = train_df["label"].tolist() + aug_labels
        print(f"  Original: {len(train_df)}, Augmented: {len(aug_texts)}, "
              f"Combined: {len(combined_texts)}")

        aug_df = pd.DataFrame({"text": combined_texts, "label": combined_labels})
        train_dataset_aug = df_to_dataset(aug_df, tokenizer, has_labels=True)

        results["aug"] = train_and_evaluate(
            train_dataset=train_dataset_aug,
            val_dataset=val_dataset,
            val_texts=val_texts,
            run_name="With EDA Augmentation",
            output_dir=BASE_DIR / "run_aug",
            diagram_dir=DIAGRAM_DIR,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Ablation comparison (if both runs completed)
    # ------------------------------------------------------------------
    if "no_aug" in results and "aug" in results:
        print("\nGenerating ablation comparison...")
        plot_ablation_comparison(
            results["no_aug"]["metrics"],
            results["aug"]["metrics"],
            DIAGRAM_DIR / "ablation_comparison.png",
        )

        # Summary table
        summary = pd.DataFrame({
            "Metric": list(results["no_aug"]["metrics"].keys()),
            "Without Augmentation": list(results["no_aug"]["metrics"].values()),
            "With EDA Augmentation": list(results["aug"]["metrics"].values()),
        })
        summary["Difference"] = summary["With EDA Augmentation"] - summary["Without Augmentation"]
        summary_path = DIAGRAM_DIR / "ablation_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")
        print("\n  Ablation Summary:")
        print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("  All done! Check the diagrams/ folder for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
