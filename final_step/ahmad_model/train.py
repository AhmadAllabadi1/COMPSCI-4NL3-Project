"""
RoBERTa Fine-Tuning for Reddit Advice Intent Classification
=============================================================
Runs three ablation experiments and produces all evaluation outputs:

  Run 1: RoBERTa + Weighted Cross-Entropy  (no augmentation)
  Run 2: RoBERTa + Focal Loss              (no augmentation)
  Run 3: RoBERTa + Focal Loss              (with targeted EDA augmentation)
  Run 4: RoBERTa + Weighted Cross-Entropy  (with targeted EDA augmentation)

Two ablation comparisons are generated:
  - Loss Function:    Weighted CE vs Focal Loss  (Runs 1 vs 2)
  - Augmentation:     No Aug vs With Aug         (Runs 2 vs 3, Runs 1 vs 4)

All outputs (plots, reports, metrics) are saved to ./diagrams/

Usage:
    python train.py                 # run all 3 experiments
    python train.py --run 1         # run only experiment 1
    python train.py --run 2         # run only experiment 2
    python train.py --run 3         # run only experiment 3
    python train.py --run 1 2       # run experiments 1 and 2
"""

import argparse
import json
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from eda_augment import augment_minority_classes

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
DIAGRAM_DIR = BASE_DIR / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_NAME = "roberta-base"
MAX_LEN = 512
SEED = 42

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
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples so the model focuses on hard, misclassified ones.
    gamma=0 reduces to standard cross-entropy.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        focal_weight = (1 - target_probs) ** self.gamma
        if self.alpha is not None:
            alpha_weight = self.alpha.to(logits.device).gather(0, targets)
            focal_weight = focal_weight * alpha_weight
        return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# Custom Trainers
# ---------------------------------------------------------------------------
class WeightedCETrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


class FocalTrainer(Trainer):
    """Trainer with Focal Loss."""

    def __init__(self, focal_loss: FocalLoss = None, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels) if self.focal_loss else torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data(path: Path, has_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).fillna("")
    if has_labels:
        df["label"] = df["label"].str.strip().str.upper()
    return df


def df_to_dataset(df: pd.DataFrame, tokenizer, has_labels: bool = True) -> Dataset:
    data_dict = {"text": df["text"].tolist()}
    if has_labels:
        data_dict["label"] = [LABEL2ID[l] for l in df["label"]]
    dataset = Dataset.from_dict(data_dict)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return dataset


def compute_class_weights(labels: list) -> torch.Tensor:
    """Inverse-frequency weights: total / (num_classes * count_per_class)."""
    counts = Counter(labels)
    total = sum(counts.values())
    n = len(counts)
    weights = torch.zeros(n)
    for label_id, count in counts.items():
        weights[label_id] = total / (n * count)
    return weights


def compute_eval_metrics(eval_pred):
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
# Visualization
# ---------------------------------------------------------------------------
def plot_label_distribution(train_df, val_df, test_df, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
        if "label" in df.columns:
            counts = df["label"].value_counts().reindex(LABEL_LIST, fill_value=0)
            colors = sns.color_palette("viridis", NUM_LABELS)
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
    path = save_dir / "label_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
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


def plot_loss_curves(log_history, save_path, title):
    train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_loss = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        ep, lo = zip(*train_loss)
        ax.plot(ep, lo, "o-", label="Train Loss", markersize=4)
    if eval_loss:
        ep, lo = zip(*eval_loss)
        ax.plot(ep, lo, "s-", label="Validation Loss", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_classification_report_file(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0)
    save_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {save_path}")
    return report


def save_misclassified(texts, y_true, y_pred, save_path, max_samples=50):
    rows = []
    for text, t, p in zip(texts, y_true, y_pred):
        if t != p:
            rows.append({"text": text, "true_label": LABEL_LIST[t], "predicted_label": LABEL_LIST[p]})
    df = pd.DataFrame(rows[:max_samples])
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path} ({len(df)} examples)")


def plot_ablation(metrics_a, metrics_b, label_a, label_b, title, save_path):
    """Side-by-side bar chart comparing two runs."""
    names = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    vals_a = [metrics_a[m] for m in names]
    vals_b = [metrics_b[m] for m in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, vals_a, w, label=label_a, color="#4C72B0")
    b2 = ax.bar(x + w / 2, vals_b, w, label=label_b, color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in names], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_runs_comparison(all_results, save_path):
    """Grouped bar chart comparing all 3 runs across all metrics."""
    names = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    display = [m.replace("_", " ").title() for m in names]
    run_labels = list(all_results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(names))
    n = len(run_labels)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (label, res) in enumerate(all_results.items()):
        vals = [res["metrics"][m] for m in names]
        bars = ax.bar(x + (i - n / 2 + 0.5) * w, vals, w, label=label, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Runs Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1_comparison(all_results, save_path):
    """Grouped bar chart comparing per-class F1 across all runs."""
    run_labels = list(all_results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(NUM_LABELS)
    n = len(run_labels)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (label, res) in enumerate(all_results.items()):
        y_true = res["true_labels"]
        y_pred = res["predictions"]
        report = classification_report(y_true, y_pred, target_names=LABEL_LIST,
                                       output_dict=True, zero_division=0)
        f1s = [report[cls]["f1-score"] for cls in LABEL_LIST]
        bars = ax.bar(x + (i - n / 2 + 0.5) * w, f1s, w, label=label, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Comparison Across All Runs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_LIST, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_experiment(
    run_tag: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    val_texts: list,
    trainer_cls,
    trainer_extra_kwargs: dict,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
) -> dict:
    """Run one training experiment and save all outputs."""

    print(f"\n{'='*60}")
    print(f"  Run: {run_tag}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"{'='*60}\n")

    # Fresh model each run
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=0.3,
    )

    output_dir = BASE_DIR / f"run_{run_tag.lower().replace(' ', '_').replace('+', '').replace('  ', '_')}"

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
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = trainer_cls(
        **trainer_extra_kwargs,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_eval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n  Results for {run_tag}:")
    for k, v in eval_results.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Predictions
    predictions = trainer.predict(val_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = np.array(val_dataset["label"])

    # File prefix for this run
    prefix = run_tag.lower().replace(" ", "_").replace("+", "").replace("  ", "_")

    # Save all per-run outputs
    plot_confusion_matrix(true_ids, pred_ids,
                          DIAGRAM_DIR / f"{prefix}_confusion_matrix.png",
                          f"Confusion Matrix — {run_tag}")

    report = save_classification_report_file(true_ids, pred_ids,
                                             DIAGRAM_DIR / f"{prefix}_classification_report.txt")
    print(f"\n{report}")

    plot_loss_curves(trainer.state.log_history,
                     DIAGRAM_DIR / f"{prefix}_loss_curves.png",
                     f"Loss Curves — {run_tag}")

    save_misclassified(val_texts, true_ids, pred_ids,
                       DIAGRAM_DIR / f"{prefix}_misclassified.csv")

    metrics = {
        "accuracy": accuracy_score(true_ids, pred_ids),
        "f1_macro": f1_score(true_ids, pred_ids, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "precision_macro": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "recall_macro": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
    }
    metrics_path = DIAGRAM_DIR / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved: {metrics_path}")

    # Save best model
    trainer.save_model(str(output_dir / "best_model"))
    print(f"  Model saved to: {output_dir / 'best_model'}")

    return {
        "metrics": metrics,
        "predictions": pred_ids,
        "true_labels": true_ids,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RoBERTa ablation study for advice intent classification")
    parser.add_argument("--run", type=int, nargs="*", default=None,
                        help="Which runs to execute (1, 2, 3, 4). Default: all.")
    parser.add_argument("--eda-alpha", type=float, default=0.1, help="EDA intensity (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    args = parser.parse_args()

    runs_to_do = set(args.run) if args.run else {1, 2, 3, 4}

    set_seed(SEED)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Loading data")
    print("=" * 60)
    train_df = load_data(TRAIN_CSV, has_labels=True)
    val_df = load_data(VAL_CSV, has_labels=True)
    test_df = load_data(TEST_CSV, has_labels=False)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Label distributions
    plot_label_distribution(train_df, val_df, test_df, DIAGRAM_DIR)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Shared validation set
    val_dataset = df_to_dataset(val_df, tokenizer, has_labels=True)
    val_texts = val_df["text"].tolist()

    # Class weights
    train_label_ids = [LABEL2ID[l] for l in train_df["label"]]
    class_weights = compute_class_weights(train_label_ids)
    print(f"\n  Class weights: { {LABEL_LIST[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())} }")

    # Tokenize original training set (shared by runs 1 & 2)
    train_dataset = df_to_dataset(train_df, tokenizer, has_labels=True)

    all_results = {}

    # ------------------------------------------------------------------
    # Run 1: RoBERTa + Weighted CE (no augmentation)
    # ------------------------------------------------------------------
    if 1 in runs_to_do:
        all_results["Weighted CE"] = run_experiment(
            run_tag="Weighted CE",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_texts=val_texts,
            trainer_cls=WeightedCETrainer,
            trainer_extra_kwargs={"class_weights": class_weights},
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Run 2: RoBERTa + Focal Loss (no augmentation)
    # ------------------------------------------------------------------
    if 2 in runs_to_do:
        focal = FocalLoss(alpha=class_weights, gamma=args.gamma)
        all_results["Focal Loss"] = run_experiment(
            run_tag="Focal Loss",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_texts=val_texts,
            trainer_cls=FocalTrainer,
            trainer_extra_kwargs={"focal_loss": focal},
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Run 3: RoBERTa + Focal Loss + Targeted EDA (with augmentation)
    # ------------------------------------------------------------------
    if 3 in runs_to_do:
        print(f"\nApplying targeted EDA augmentation (minority → majority count, alpha={args.eda_alpha})...")
        aug_texts, aug_labels = augment_minority_classes(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            target_count=None,
            alpha=args.eda_alpha,
        )
        combined_texts = train_df["text"].tolist() + aug_texts
        combined_labels = train_df["label"].tolist() + aug_labels
        new_dist = Counter(combined_labels)
        print(f"  Original: {len(train_df)}, Augmented: {len(aug_texts)}, Combined: {len(combined_texts)}")
        print(f"  Distribution: {dict(sorted(new_dist.items()))}")

        aug_df = pd.DataFrame({"text": combined_texts, "label": combined_labels})
        aug_dataset = df_to_dataset(aug_df, tokenizer, has_labels=True)

        aug_label_ids = [LABEL2ID[l] for l in combined_labels]
        aug_weights = compute_class_weights(aug_label_ids)
        focal_aug = FocalLoss(alpha=aug_weights, gamma=args.gamma)

        all_results["Focal Loss + EDA"] = run_experiment(
            run_tag="Focal Loss + EDA",
            train_dataset=aug_dataset,
            val_dataset=val_dataset,
            val_texts=val_texts,
            trainer_cls=FocalTrainer,
            trainer_extra_kwargs={"focal_loss": focal_aug},
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Run 4: RoBERTa + Weighted CE + Targeted EDA (with augmentation)
    # ------------------------------------------------------------------
    if 4 in runs_to_do:
        # Reuse augmented data from Run 3 if available, otherwise generate it
        if "aug_dataset" not in locals():
            print(f"\nApplying targeted EDA augmentation (minority → majority count, alpha={args.eda_alpha})...")
            aug_texts, aug_labels = augment_minority_classes(
                train_df["text"].tolist(),
                train_df["label"].tolist(),
                target_count=None,
                alpha=args.eda_alpha,
            )
            combined_texts = train_df["text"].tolist() + aug_texts
            combined_labels = train_df["label"].tolist() + aug_labels
            new_dist = Counter(combined_labels)
            print(f"  Original: {len(train_df)}, Augmented: {len(aug_texts)}, Combined: {len(combined_texts)}")
            print(f"  Distribution: {dict(sorted(new_dist.items()))}")

            aug_df = pd.DataFrame({"text": combined_texts, "label": combined_labels})
            aug_dataset = df_to_dataset(aug_df, tokenizer, has_labels=True)

            aug_label_ids = [LABEL2ID[l] for l in combined_labels]
            aug_weights = compute_class_weights(aug_label_ids)

        all_results["Weighted CE + EDA"] = run_experiment(
            run_tag="Weighted CE + EDA",
            train_dataset=aug_dataset,
            val_dataset=val_dataset,
            val_texts=val_texts,
            trainer_cls=WeightedCETrainer,
            trainer_extra_kwargs={"class_weights": aug_weights},
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Ablation comparisons
    # ------------------------------------------------------------------

    # Loss function ablation (no augmentation): Weighted CE vs Focal Loss
    if "Weighted CE" in all_results and "Focal Loss" in all_results:
        print("\n  Generating loss function ablation...")
        plot_ablation(
            all_results["Weighted CE"]["metrics"],
            all_results["Focal Loss"]["metrics"],
            "Weighted CE", "Focal Loss",
            "Ablation: Loss Function (Weighted CE vs Focal Loss)",
            DIAGRAM_DIR / "ablation_loss_function.png",
        )

    # Augmentation ablation (Focal Loss): No Aug vs EDA
    if "Focal Loss" in all_results and "Focal Loss + EDA" in all_results:
        print("\n  Generating Focal Loss augmentation ablation...")
        plot_ablation(
            all_results["Focal Loss"]["metrics"],
            all_results["Focal Loss + EDA"]["metrics"],
            "No Augmentation", "With EDA",
            "Ablation: Augmentation with Focal Loss",
            DIAGRAM_DIR / "ablation_augmentation_focal.png",
        )

    # Augmentation ablation (Weighted CE): No Aug vs EDA
    if "Weighted CE" in all_results and "Weighted CE + EDA" in all_results:
        print("\n  Generating Weighted CE augmentation ablation...")
        plot_ablation(
            all_results["Weighted CE"]["metrics"],
            all_results["Weighted CE + EDA"]["metrics"],
            "No Augmentation", "With EDA",
            "Ablation: Augmentation with Weighted CE",
            DIAGRAM_DIR / "ablation_augmentation_wce.png",
        )

    # ------------------------------------------------------------------
    # Overall comparison (all runs)
    # ------------------------------------------------------------------
    if len(all_results) >= 2:
        plot_all_runs_comparison(all_results, DIAGRAM_DIR / "all_runs_comparison.png")
        plot_per_class_f1_comparison(all_results, DIAGRAM_DIR / "per_class_f1_comparison.png")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if all_results:
        rows = []
        for label, res in all_results.items():
            row = {"Run": label}
            row.update(res["metrics"])
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = DIAGRAM_DIR / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}")
        print("\n  Final Summary:")
        print("  " + summary.to_string(index=False).replace("\n", "\n  "))

    print("\n" + "=" * 60)
    print("  All done! Check diagrams/ for all outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
