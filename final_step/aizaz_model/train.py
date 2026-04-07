"""
TextCNN for Reddit advice intent classification.

3 runs:
  1. Random embeddings + class weights
  2. Word2Vec init + class weights
  3. Word2Vec init + no class weights

Best run is selected by validation macro F1.
All outputs are saved to aizaz_model/diagrams/
"""

import copy
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as TF
from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
DIAGRAM_DIR = BASE_DIR / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

SEED = 42
DEVICE = torch.device("cpu")

LABEL_LIST = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

EMBED_DIM = 200
MAX_VOCAB = 30000
MIN_FREQ = 2
MAX_LEN_CAP = 256
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 12
LR = 1e-3
PATIENCE = 3
NUM_FILTERS = 100
KERNEL_SIZES = (3, 4, 5)
DROPOUT = 0.5

RUNS = [
    {"name": "textcnn_random_weighted", "use_w2v": False, "use_class_weights": True, "freeze_embeddings": False},
    {"name": "textcnn_w2v_weighted", "use_w2v": True, "use_class_weights": True, "freeze_embeddings": False},
    {"name": "textcnn_w2v_unweighted", "use_w2v": True, "use_class_weights": False, "freeze_embeddings": False},
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)
    return df


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocab(tokenized_texts, min_freq=MIN_FREQ, max_size=MAX_VOCAB):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, count in counter.most_common():
        if len(vocab) >= max_size:
            break
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab, max_len):
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([PAD_IDX] * (max_len - len(ids)))
    return ids


def choose_max_len(tokenized_texts, quantile=0.95, hard_cap=MAX_LEN_CAP):
    lengths = np.array([len(tokens) for tokens in tokenized_texts], dtype=np.int32)
    if len(lengths) == 0:
        return 32
    q_len = int(np.quantile(lengths, quantile))
    return max(16, min(hard_cap, q_len))


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_metrics(y_true_ids, y_pred_ids):
    y_true = [ID2LABEL[i] for i in y_true_ids]
    y_pred = [ID2LABEL[i] for i in y_pred_ids]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_LIST, zero_division=0
    )

    per_class = {}
    for label, p, r, f, s in zip(LABEL_LIST, precision, recall, f1, support):
        per_class[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_LIST, output_dict=True, zero_division=0
        ),
    }


# Plotting helpers

def plot_label_distribution(train_df, val_df, test_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
        counts = df["label"].value_counts().reindex(LABEL_LIST, fill_value=0)
        bars = ax.bar(LABEL_LIST, counts.values, color=sns.color_palette("muted", NUM_LABELS))
        ax.bar_label(bars, fontsize=8)
        ax.set_title(f"{name} Set (n={len(df)})", fontsize=13)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)
    path = DIAGRAM_DIR / "label_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true_ids, y_pred_ids, save_path, title):
    y_true = [ID2LABEL[i] for i in y_true_ids]
    y_pred = [ID2LABEL[i] for i in y_pred_ids]
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


def plot_loss_curves(history, save_path, title):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", label="Validation Loss", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_macro_f1"], "^-", color="#C44E52", label="Validation Macro F1", markersize=4)
    ax2.set_ylabel("Macro F1", fontsize=12)
    ax2.set_ylim(0.0, 1.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    ax1.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(metrics, save_path, title):
    vals = [metrics["per_class"][label]["f1"] for label in LABEL_LIST]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(LABEL_LIST, vals, color=sns.color_palette("deep", NUM_LABELS))
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_run_comparison(all_results, split_name, save_path):
    metric_names = ["accuracy", "f1_macro", "f1_weighted"]
    run_names = list(all_results.keys())
    x = np.arange(len(metric_names))
    w = 0.8 / len(run_names)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, run_name in enumerate(run_names):
        vals = [all_results[run_name][split_name][m] for m in metric_names]
        bars = ax.bar(x + (i - len(run_names) / 2 + 0.5) * w, vals, w, label=run_name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"TextCNN Ablation Comparison ({split_name.title()})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_classification_report_file(metrics, save_path):
    lines = []
    for label in LABEL_LIST:
        row = metrics["per_class"][label]
        lines.append(
            f"{label:20s}  precision={row['precision']:.4f}  recall={row['recall']:.4f}  "
            f"f1={row['f1']:.4f}  support={row['support']}"
        )
    lines.append("")
    lines.append(f"accuracy     = {metrics['accuracy']:.4f}")
    lines.append(f"macro_f1     = {metrics['f1_macro']:.4f}")
    lines.append(f"weighted_f1  = {metrics['f1_weighted']:.4f}")
    lines.append(f"macro_prec   = {metrics['precision_macro']:.4f}")
    lines.append(f"macro_recall = {metrics['recall_macro']:.4f}")
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {save_path}")


# Embeddings

def train_word2vec(tokenized_texts):
    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=EMBED_DIM,
        window=5,
        min_count=MIN_FREQ,
        workers=1,
        sg=1,
        epochs=25,
        seed=SEED,
    )


def build_embedding_matrix(vocab, w2v_model=None):
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0.0, 0.05, size=(len(vocab), EMBED_DIM)).astype(np.float32)
    matrix[PAD_IDX] = np.zeros(EMBED_DIM, dtype=np.float32)

    if w2v_model is None:
        return matrix

    for token, idx in vocab.items():
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if token in w2v_model.wv:
            matrix[idx] = w2v_model.wv[token]
    return matrix


# Dataset / model

class TextDataset(Dataset):
    def __init__(self, encoded_texts, labels):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, k) for k in KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), NUM_LABELS)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            out = TF.relu(conv(x))
            pooled.append(torch.max(out, dim=2).values)
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.classifier(x)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, y_true, y_pred


def train_one_run(run_cfg, train_loader, val_loader, test_loader, embedding_matrix, class_weights):
    model = TextCNN(
        vocab_size=embedding_matrix.shape[0],
        embedding_matrix=embedding_matrix,
        freeze_embeddings=run_cfg["freeze_embeddings"],
    ).to(DEVICE)

    weight_tensor = None
    if run_cfg["use_class_weights"]:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_f1 = -math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    print(f"\nRunning {run_cfg['name']}")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

        train_loss = total_train_loss / max(1, len(train_loader.dataset))
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(float(val_f1))

        print(
            f"  epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_loss, val_true, val_pred = evaluate(model, val_loader, criterion)
    test_loss, test_true, test_pred = evaluate(model, test_loader, criterion)

    return {
        "run_name": run_cfg["name"],
        "best_epoch": best_epoch,
        "history": history,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_true": val_true,
        "val_pred": val_pred,
        "test_true": test_true,
        "test_pred": test_pred,
        "val_metrics": compute_metrics(val_true, val_pred),
        "test_metrics": compute_metrics(test_true, test_pred),
    }


def save_predictions(df, y_true_ids, y_pred_ids, save_path):
    out = pd.DataFrame({
        "id": df["id"],
        "text": df["text"],
        "true_label": [ID2LABEL[i] for i in y_true_ids],
        "predicted_label": [ID2LABEL[i] for i in y_pred_ids],
    })
    out.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")
    return out


def save_run_outputs(result, val_df, test_df, metadata):
    run_name = result["run_name"]

    save_json(
        DIAGRAM_DIR / f"{run_name}_val_metrics.json",
        {
            "run": run_name,
            "best_epoch": result["best_epoch"],
            "loss": result["val_loss"],
            **result["val_metrics"],
            **metadata,
        },
    )
    save_json(
        DIAGRAM_DIR / f"{run_name}_test_metrics.json",
        {
            "run": run_name,
            "best_epoch": result["best_epoch"],
            "loss": result["test_loss"],
            **result["test_metrics"],
            **metadata,
        },
    )

    save_classification_report_file(
        result["val_metrics"],
        DIAGRAM_DIR / f"{run_name}_val_classification_report.txt",
    )
    save_classification_report_file(
        result["test_metrics"],
        DIAGRAM_DIR / f"{run_name}_test_classification_report.txt",
    )

    plot_confusion_matrix(
        result["val_true"], result["val_pred"],
        DIAGRAM_DIR / f"{run_name}_val_confusion_matrix.png",
        f"{run_name} Validation Confusion Matrix",
    )
    plot_confusion_matrix(
        result["test_true"], result["test_pred"],
        DIAGRAM_DIR / f"{run_name}_test_confusion_matrix.png",
        f"{run_name} Test Confusion Matrix",
    )
    plot_loss_curves(
        result["history"],
        DIAGRAM_DIR / f"{run_name}_loss_curves.png",
        f"{run_name} Loss Curves",
    )
    plot_per_class_f1(
        result["test_metrics"],
        DIAGRAM_DIR / f"{run_name}_test_per_class_f1.png",
        f"{run_name} Test Per-Class F1",
    )

    save_predictions(
        val_df, result["val_true"], result["val_pred"],
        DIAGRAM_DIR / f"{run_name}_val_predictions.csv",
    )
    test_pred_df = save_predictions(
        test_df, result["test_true"], result["test_pred"],
        DIAGRAM_DIR / f"{run_name}_test_predictions.csv",
    )

    test_mis = test_pred_df[test_pred_df["true_label"] != test_pred_df["predicted_label"]].copy()
    test_mis.to_csv(DIAGRAM_DIR / f"{run_name}_test_misclassified.csv", index=False)
    print(f"  Saved: {DIAGRAM_DIR / f'{run_name}_test_misclassified.csv'}")


def main():
    set_seed(SEED)

    print("=" * 60)
    print("  Loading data")
    print("=" * 60)
    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)
    test_df = load_data(TEST_CSV)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    plot_label_distribution(train_df, val_df, test_df)

    tokenized_train = [tokenize(text) for text in train_df["text"]]
    tokenized_val = [tokenize(text) for text in val_df["text"]]
    tokenized_test = [tokenize(text) for text in test_df["text"]]

    vocab = build_vocab(tokenized_train)
    max_len = choose_max_len(tokenized_train)

    print(f"\n  Vocabulary size: {len(vocab)}")
    print(f"  Max length: {max_len}")
    print(f"  Device: {DEVICE}")

    X_train = [encode_tokens(tokens, vocab, max_len) for tokens in tokenized_train]
    X_val = [encode_tokens(tokens, vocab, max_len) for tokens in tokenized_val]
    X_test = [encode_tokens(tokens, vocab, max_len) for tokens in tokenized_test]

    y_train = [LABEL2ID[label] for label in train_df["label"]]
    y_val = [LABEL2ID[label] for label in val_df["label"]]
    y_test = [LABEL2ID[label] for label in test_df["label"]]

    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=EVAL_BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=np.array(y_train),
    ).astype(np.float32)
    class_weight_map = {label: float(w) for label, w in zip(LABEL_LIST, class_weights)}
    print(f"\n  Class weights: {class_weight_map}")

    print("\nTraining Word2Vec on training split...")
    w2v_model = train_word2vec(tokenized_train)

    metadata = {
        "seed": SEED,
        "vocab_size": len(vocab),
        "max_length": max_len,
        "embed_dim": EMBED_DIM,
        "class_weights": class_weight_map,
    }
    save_json(DIAGRAM_DIR / "run_metadata.json", metadata)

    all_results = {}
    for run_cfg in RUNS:
        embedding_matrix = build_embedding_matrix(
            vocab,
            w2v_model if run_cfg["use_w2v"] else None,
        )
        result = train_one_run(
            run_cfg,
            train_loader,
            val_loader,
            test_loader,
            embedding_matrix,
            class_weights,
        )
        save_run_outputs(result, val_df, test_df, metadata)
        all_results[run_cfg["name"]] = {
            "val": result["val_metrics"],
            "test": result["test_metrics"],
            "best_epoch": result["best_epoch"],
            "use_w2v_init": run_cfg["use_w2v"],
            "use_class_weights": run_cfg["use_class_weights"],
            "freeze_embeddings": run_cfg["freeze_embeddings"],
        }

    val_rows = []
    test_rows = []
    for run_name, result in all_results.items():
        val_rows.append({
            "Run": run_name,
            "accuracy": result["val"]["accuracy"],
            "f1_macro": result["val"]["f1_macro"],
            "f1_weighted": result["val"]["f1_weighted"],
            "precision_macro": result["val"]["precision_macro"],
            "recall_macro": result["val"]["recall_macro"],
            "best_epoch": result["best_epoch"],
            "use_w2v_init": result["use_w2v_init"],
            "use_class_weights": result["use_class_weights"],
        })
        test_rows.append({
            "Run": run_name,
            "accuracy": result["test"]["accuracy"],
            "f1_macro": result["test"]["f1_macro"],
            "f1_weighted": result["test"]["f1_weighted"],
            "precision_macro": result["test"]["precision_macro"],
            "recall_macro": result["test"]["recall_macro"],
            "best_epoch": result["best_epoch"],
            "use_w2v_init": result["use_w2v_init"],
            "use_class_weights": result["use_class_weights"],
        })

    val_summary = pd.DataFrame(val_rows)
    test_summary = pd.DataFrame(test_rows)
    val_summary.to_csv(DIAGRAM_DIR / "val_summary.csv", index=False)
    test_summary.to_csv(DIAGRAM_DIR / "test_summary.csv", index=False)
    print(f"\n  Saved: {DIAGRAM_DIR / 'val_summary.csv'}")
    print(f"  Saved: {DIAGRAM_DIR / 'test_summary.csv'}")

    plot_run_comparison(all_results, "val", DIAGRAM_DIR / "ablation_validation_comparison.png")
    plot_run_comparison(all_results, "test", DIAGRAM_DIR / "ablation_test_comparison.png")

    best_run_name = max(all_results, key=lambda name: all_results[name]["val"]["f1_macro"])
    best_result = all_results[best_run_name]

    save_json(
        DIAGRAM_DIR / "best_run_summary.json",
        {
            "best_run": best_run_name,
            "selection_metric": "validation_macro_f1",
            "validation_metrics": best_result["val"],
            "test_metrics": best_result["test"],
            "best_epoch": best_result["best_epoch"],
            "config": {
                "use_w2v_init": best_result["use_w2v_init"],
                "use_class_weights": best_result["use_class_weights"],
                "freeze_embeddings": best_result["freeze_embeddings"],
            },
        },
    )

    print("\n" + "=" * 60)
    print(f"  Best run by val macro F1: {best_run_name}")
    print("=" * 60)
    print(
        f"  Validation: accuracy={best_result['val']['accuracy']:.4f}, "
        f"macro_f1={best_result['val']['f1_macro']:.4f}, "
        f"weighted_f1={best_result['val']['f1_weighted']:.4f}"
    )
    print(
        f"  Test:       accuracy={best_result['test']['accuracy']:.4f}, "
        f"macro_f1={best_result['test']['f1_macro']:.4f}, "
        f"weighted_f1={best_result['test']['f1_weighted']:.4f}"
    )
    print(f"  Outputs saved to: {DIAGRAM_DIR}")


if __name__ == "__main__":
    main()
