"""
RoBERTa Fine-Tuning with Focal Loss for Reddit Advice Intent Classification
=============================================================================
- Uses roberta-base instead of DistilBERT for stronger pretrained representations
- Focal Loss instead of weighted cross-entropy for better handling of class imbalance
- Runs ablation: without augmentation vs with targeted minority-class EDA augmentation
- All outputs saved to ./diagrams/

Usage:
    python train_roberta.py                    # full ablation (no-aug + aug)
    python train_roberta.py --no-augmentation  # baseline only
    python train_roberta.py --augmentation     # augmented only
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
import torch.nn.functional as F
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
    Focal Loss (Lin et al., 2017) for imbalanced classification.

    Instead of standard cross-entropy:
        CE(p) = -log(p)

    Focal loss adds a modulating factor:
        FL(p) = -alpha * (1 - p)^gamma * log(p)

    - When the model is confident and correct (p is high), (1-p)^gamma → 0,
      so the loss is down-weighted. Easy examples contribute less.
    - When the model is wrong (p is low), (1-p)^gamma → 1, so the loss stays
      high. Hard examples dominate the gradient.

    Args:
        alpha: Per-class weight tensor (like class weights in weighted CE).
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is standard cross-entropy. gamma=2 is the paper default.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        # Gather the probability assigned to the correct class for each sample
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Standard CE component: -log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha.to(logits.device).gather(0, targets)
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * ce_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Data Loading
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
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return dataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def compute_class_weights(labels: list) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.zeros(num_classes)
    for label_id, count in counts.items():
        weights[label_id] = total / (num_classes * count)
    return weights


# ---------------------------------------------------------------------------
# Custom Trainer with Focal Loss
# ---------------------------------------------------------------------------
class FocalTrainer(Trainer):
    """Trainer that uses Focal Loss instead of standard cross-entropy."""

    def __init__(self, focal_loss: FocalLoss = None, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.focal_loss is not None:
            loss = self.focal_loss(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_label_distribution(train_df, val_df, test_df, save_dir):
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


def plot_confusion_matrix(y_true, y_pred, label_names, save_path, title="Confusion Matrix"):
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


def plot_training_curves(log_history, save_path, title="Training & Validation Loss"):
    train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_loss = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
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


def save_classification_report(y_true, y_pred, label_names, save_path):
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    save_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {save_path}")
    return report


def save_misclassified(texts, y_true, y_pred, label_names, save_path, max_samples=50):
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
    focal_loss: FocalLoss,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"  Loss: Focal Loss (gamma={focal_loss.gamma})")
    print(f"{'='*60}\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=0.3,
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
        warmup_ratio=0.1,  # warm up LR over first 10% of steps
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = FocalTrainer(
        focal_loss=focal_loss,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n  Eval results for {run_name}:")
    for k, v in eval_results.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    predictions = trainer.predict(val_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = np.array(val_dataset["label"])

    # --- Save outputs ---
    prefix = run_name.lower().replace(" ", "_").replace("+", "").replace("  ", "_")

    plot_confusion_matrix(
        true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"roberta_{prefix}_confusion_matrix.png",
        title=f"Confusion Matrix — RoBERTa {run_name}",
    )

    report_str = save_classification_report(
        true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"roberta_{prefix}_classification_report.txt",
    )
    print(f"\n  Classification Report ({run_name}):\n{report_str}")

    plot_training_curves(
        trainer.state.log_history,
        diagram_dir / f"roberta_{prefix}_loss_curves.png",
        title=f"Loss Curves — RoBERTa {run_name}",
    )

    save_misclassified(
        val_texts, true_ids, pred_ids, LABEL_LIST,
        diagram_dir / f"roberta_{prefix}_misclassified.csv",
    )

    metrics = {
        "accuracy": accuracy_score(true_ids, pred_ids),
        "f1_macro": f1_score(true_ids, pred_ids, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "precision_macro": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "recall_macro": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
    }
    metrics_path = diagram_dir / f"roberta_{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved: {metrics_path}")

    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    print(f"  Best model saved to: {best_model_dir}")

    return {"metrics": metrics, "predictions": pred_ids, "true_labels": true_ids}


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------
def plot_ablation_comparison(metrics_no_aug, metrics_aug, save_path):
    metric_names = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    no_aug_vals = [metrics_no_aug[m] for m in metric_names]
    aug_vals = [metrics_aug[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, no_aug_vals, width, label="Without Augmentation", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, aug_vals, width, label="With EDA Augmentation", color="#DD8452")
    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("RoBERTa + Focal Loss: With vs Without EDA Augmentation", fontsize=14)
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
    parser = argparse.ArgumentParser(description="RoBERTa + Focal Loss fine-tuning")
    parser.add_argument("--no-augmentation", action="store_true", help="Only run without augmentation")
    parser.add_argument("--augmentation", action="store_true", help="Only run with augmentation")
    parser.add_argument("--eda-alpha", type=float, default=0.1, help="EDA intensity (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    args = parser.parse_args()

    set_seed(SEED)

    run_no_aug = not args.augmentation
    run_aug = not args.no_augmentation

    # Load data
    print("Loading data...")
    train_df = load_data(TRAIN_CSV, has_labels=True)
    val_df = load_data(VAL_CSV, has_labels=True)
    test_df = load_data(TEST_CSV, has_labels=False)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Label distribution
    print("\nPlotting label distributions...")
    plot_label_distribution(train_df, val_df, test_df, DIAGRAM_DIR)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    val_dataset = df_to_dataset(val_df, tokenizer, has_labels=True)
    val_texts = val_df["text"].tolist()

    # Class weights for focal loss alpha
    train_label_ids = [LABEL2ID[l] for l in train_df["label"]]
    alpha_weights = compute_class_weights(train_label_ids)
    print(f"\n  Class weights (focal alpha): {dict(zip(LABEL_LIST, alpha_weights.tolist()))}")

    results = {}

    # ------------------------------------------------------------------
    # Run 1: No augmentation
    # ------------------------------------------------------------------
    if run_no_aug:
        focal = FocalLoss(alpha=alpha_weights, gamma=args.gamma)
        train_dataset_no_aug = df_to_dataset(train_df, tokenizer, has_labels=True)
        results["no_aug"] = train_and_evaluate(
            train_dataset=train_dataset_no_aug,
            val_dataset=val_dataset,
            val_texts=val_texts,
            run_name="No Augmentation",
            output_dir=BASE_DIR / "roberta_run_no_aug",
            diagram_dir=DIAGRAM_DIR,
            focal_loss=focal,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Run 2: Targeted minority augmentation
    # ------------------------------------------------------------------
    if run_aug:
        print(f"\nApplying targeted EDA augmentation (minority classes only, alpha={args.eda_alpha})...")
        aug_texts, aug_labels = augment_minority_classes(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            target_count=None,
            alpha=args.eda_alpha,
        )
        combined_texts = train_df["text"].tolist() + aug_texts
        combined_labels = train_df["label"].tolist() + aug_labels
        print(f"  Original: {len(train_df)}, Augmented (minority only): {len(aug_texts)}, "
              f"Combined: {len(combined_texts)}")
        new_dist = Counter(combined_labels)
        print(f"  New distribution: {dict(sorted(new_dist.items()))}")

        aug_df = pd.DataFrame({"text": combined_texts, "label": combined_labels})
        train_dataset_aug = df_to_dataset(aug_df, tokenizer, has_labels=True)

        # Recompute weights for balanced dataset
        aug_label_ids = [LABEL2ID[l] for l in combined_labels]
        aug_alpha = compute_class_weights(aug_label_ids)

        focal_aug = FocalLoss(alpha=aug_alpha, gamma=args.gamma)
        results["aug"] = train_and_evaluate(
            train_dataset=train_dataset_aug,
            val_dataset=val_dataset,
            val_texts=val_texts,
            run_name="With EDA Augmentation",
            output_dir=BASE_DIR / "roberta_run_aug",
            diagram_dir=DIAGRAM_DIR,
            focal_loss=focal_aug,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Ablation comparison
    # ------------------------------------------------------------------
    if "no_aug" in results and "aug" in results:
        print("\nGenerating ablation comparison...")
        plot_ablation_comparison(
            results["no_aug"]["metrics"],
            results["aug"]["metrics"],
            DIAGRAM_DIR / "roberta_ablation_comparison.png",
        )
        summary = pd.DataFrame({
            "Metric": list(results["no_aug"]["metrics"].keys()),
            "Without Augmentation": list(results["no_aug"]["metrics"].values()),
            "With EDA Augmentation": list(results["aug"]["metrics"].values()),
        })
        summary["Difference"] = summary["With EDA Augmentation"] - summary["Without Augmentation"]
        summary_path = DIAGRAM_DIR / "roberta_ablation_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")
        print("\n  Ablation Summary:")
        print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("  All done! Check the diagrams/ folder for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
