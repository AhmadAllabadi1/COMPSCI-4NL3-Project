"""
Generate CodaBench submission for Run 1: RoBERTa + Weighted Cross-Entropy (no EDA).

Usage:
    python make_submission.py                   # train from scratch
    python make_submission.py --model-dir ./run_weighted_ce/best_model  # load saved model

Output: submission.zip containing submission.csv with columns: id, label
Labels are mapped to CodaBench format: Advice, Anecdote, Appraisal, Emotional Support, Warning
"""

import argparse
import random
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV   = DATA_DIR / "validation.csv"
TEST_CSV  = DATA_DIR / "test.csv"

MODEL_NAME = "roberta-base"
MAX_LEN    = 512
SEED       = 42

# Internal labels (uppercase)
LABEL_LIST = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# CodaBench-format label mapping (uppercase with underscores)
CODABENCH_LABEL = {
    "ADVICE":            "ADVICE",
    "ANECDOTE":          "ANECDOTE",
    "APPRAISAL":         "APPRAISAL",
    "EMOTIONAL_SUPPORT": "EMOTIONAL_SUPPORT",
    "WARNING":           "WARNING",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Weighted CE trainer
# ---------------------------------------------------------------------------
class WeightedCETrainer(Trainer):
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data(path: Path, has_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
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
        "accuracy":         accuracy_score(labels, preds),
        "f1_macro":         f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":      f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro":  precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":     recall_score(labels, preds, average="macro",    zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to a saved best_model directory (skips training)")
    parser.add_argument("--epochs",     type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=8)
    args = parser.parse_args()

    set_seed(SEED)

    print("Loading data...")
    train_df = load_data(TRAIN_CSV, has_labels=True)
    val_df   = load_data(VAL_CSV,   has_labels=True)
    test_df  = load_data(TEST_CSV,  has_labels=False)   # no labels needed for submission
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    test_ids = test_df["id"].tolist()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    val_dataset  = df_to_dataset(val_df,  tokenizer, has_labels=True)
    test_dataset = df_to_dataset(test_df, tokenizer, has_labels=False)

    # -----------------------------------------------------------------------
    # Load or train model
    # -----------------------------------------------------------------------
    if args.model_dir and Path(args.model_dir).exists():
        print(f"Loading saved model from {args.model_dir}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        # Need a trainer just for predict()
        training_args = TrainingArguments(
            output_dir=str(BASE_DIR / "tmp_predict"),
            per_device_eval_batch_size=16,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_eval_metrics,
        )
    else:
        print("Training Run 1: RoBERTa + Weighted CE (no EDA)...")
        train_label_ids = [LABEL2ID[l] for l in train_df["label"]]
        class_weights   = compute_class_weights(train_label_ids)
        print(f"  Class weights: { {LABEL_LIST[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())} }")

        train_dataset = df_to_dataset(train_df, tokenizer, has_labels=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            classifier_dropout=0.3,
        )

        output_dir = BASE_DIR / "run_weighted_ce"
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            num_train_epochs=args.epochs,
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

        trainer = WeightedCETrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_eval_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Save best model for future use
        trainer.save_model(str(output_dir / "best_model"))
        print(f"  Model saved to: {output_dir / 'best_model'}")

        eval_results = trainer.evaluate()
        print(f"\nValidation results:")
        for k, v in eval_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    # -----------------------------------------------------------------------
    # Generate predictions on test set
    # -----------------------------------------------------------------------
    print("\nGenerating test predictions...")
    test_predictions = trainer.predict(test_dataset)
    test_pred_ids    = np.argmax(test_predictions.predictions, axis=-1)

    # Map to CodaBench labels
    pred_labels = [CODABENCH_LABEL[ID2LABEL[i]] for i in test_pred_ids]

    # -----------------------------------------------------------------------
    # Save submission
    # -----------------------------------------------------------------------
    submission_df = pd.DataFrame({"id": test_ids, "label": pred_labels})
    csv_path = BASE_DIR / "submission.csv"
    zip_path = BASE_DIR / "submission.zip"

    submission_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("Label distribution:")
    print(submission_df["label"].value_counts().to_string())

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
    print(f"Saved: {zip_path}")
    print("\nDone! Upload submission.zip to CodaBench.")


if __name__ == "__main__":
    main()
