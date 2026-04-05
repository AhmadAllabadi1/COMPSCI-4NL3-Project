"""
Model: Bidirectional LSTM (BiLSTM) trained from scratch

Architecture:
  - Word-level vocabulary built from training data (no pretrained model)
  - 128-dimensional trainable word embeddings (random init)
  - 2-layer Bidirectional LSTM (hidden=256, output dim=512)
  - Run 3 adds a soft attention layer over all LSTM hidden states

3 Ablation Runs:
  Run 1: BiLSTM + Weighted Cross-Entropy                   (base)
  Run 2: BiLSTM + EDA data augmentation + Weighted CE      (+ augmentation)
  Run 3: BiLSTM + Attention + Weighted CE                  (+ attention)

All ouputs saved to rayan_model/diagrams/:
"""

import os
import re
import sys
import json
import random
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import wordnet, stopwords

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED          = 42
MAX_LEN       = 128       # max tokens per sequence (word-level)
VOCAB_MIN_FREQ = 2        # min word frequency to include in vocab
EMBED_DIM     = 128       # word embedding size
HIDDEN_DIM    = 256       # LSTM hidden size (output = 512 after bidirectional concat)
NUM_LAYERS    = 2         # LSTM layers
DROPOUT       = 0.3
BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS     = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}
ID2LABEL   = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
print(f"Device : {DEVICE}")
print(f"Model  : BiLSTM (embed={EMBED_DIM}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS})")


# ─────────────────────────────────────────────
# EDA AUGMENTATION
# ─────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(words, n):
    new_words = words.copy()
    non_stop  = [w for w in words if w.lower() not in STOP_WORDS]
    random.shuffle(non_stop)
    replaced  = 0
    for word in non_stop:
        syns = get_synonyms(word)
        if syns:
            new_words = [random.choice(syns) if w == word else w for w in new_words]
            replaced += 1
            if replaced >= n:
                break
    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        non_stop = [w for w in new_words if w.lower() not in STOP_WORDS]
        if not non_stop:
            break
        syns = get_synonyms(random.choice(non_stop))
        if syns:
            new_words.insert(random.randint(0, len(new_words)), random.choice(syns))
    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            i, j = random.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


def random_deletion(words, p):
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p]


def eda(sentence, alpha=0.1, num_aug=1):
    words     = sentence.split()
    n         = max(1, int(alpha * len(words)))
    augmented = []
    for _ in range(num_aug):
        op = random.choice(["sr", "ri", "rs", "rd"])
        if   op == "sr": aug = synonym_replacement(words, n)
        elif op == "ri": aug = random_insertion(words, n)
        elif op == "rs": aug = random_swap(words, n)
        else:            aug = random_deletion(words, alpha)
        augmented.append(" ".join(aug))
    return augmented


def augment_minority_classes(df, target_count=None):
    """Oversample minority classes with EDA until they reach target_count."""
    counts   = df["label"].value_counts()
    majority = counts.max() if target_count is None else target_count
    new_rows = []
    for label, cnt in counts.items():
        if cnt < majority:
            subset = df[df["label"] == label]
            needed = majority - cnt
            for _ in range(needed):
                row  = subset.sample(1).iloc[0]
                augs = eda(str(row["text"]), alpha=0.1, num_aug=1)
                new_rows.append({"id": row["id"], "text": augs[0], "label": label})
    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        return pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=SEED)
    return df


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_split(filename):
    path = os.path.join(BASE_DIR, filename)
    df   = pd.read_csv(path)
    df   = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.strip()
    df   = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────
def simple_tokenize(text):
    """Lowercase word-level tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=VOCAB_MIN_FREQ):
    """Build word-to-id vocabulary from a list of strings."""
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class RedditDataset(Dataset):
    def __init__(self, df, vocab, max_len=MAX_LEN):
        self.texts   = df["text"].tolist()
        self.labels  = [LABEL2ID[l] for l in df["label"].tolist()]
        self.ids     = df["id"].tolist()   # keep original IDs for CodaBench submission
        self.vocab   = vocab
        self.max_len = max_len
        self.pad_id  = vocab[PAD_TOKEN]
        self.unk_id  = vocab[UNK_TOKEN]

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = simple_tokenize(text)[: self.max_len]
        ids    = [self.vocab.get(t, self.unk_id) for t in tokens]
        length = max(len(ids), 1)   # at least 1 for pack_padded_sequence
        ids    = ids + [self.pad_id] * (self.max_len - len(ids))
        return ids, length

    def __getitem__(self, idx):
        ids, length = self.encode(self.texts[idx])
        return {
            "input_ids": torch.tensor(ids,    dtype=torch.long),
            "length":    torch.tensor(length, dtype=torch.long),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_id": self.ids[idx],
        }


# ─────────────────────────────────────────────
# MODEL — BiLSTM  (with optional attention)
# ─────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    2-layer Bidirectional LSTM for text classification.

    Without attention (Run 1 & 2):
      The final hidden states of the forward and backward LSTMs are
      concatenated and fed to a linear classifier.

    With attention (Run 3):
      A learned attention score is computed over every timestep of the
      LSTM output. The context vector (weighted sum of hidden states)
      replaces the last-hidden-state pooling, letting the model focus
      on the most informative words regardless of their position.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # attention projection (only used in Run 3)
        if use_attention:
            self.attn_proj = nn.Linear(hidden_dim * 2, 1)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids : (batch, seq_len)
        # lengths   : (batch,) — actual non-padding length of each sequence

        x = self.dropout(self.embedding(input_ids))   # (batch, seq_len, embed_dim)

        packed         = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(packed)
        output, _      = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True
        )  # (batch, seq_len, hidden*2)

        if self.use_attention:
            # ── soft attention ──
            scores  = self.attn_proj(output).squeeze(-1)          # (batch, seq_len)
            mask    = (input_ids == 0)                            # padding mask
            scores  = scores.masked_fill(mask, -1e9)
            weights = torch.softmax(scores, dim=-1)               # (batch, seq_len)
            context = (output * weights.unsqueeze(-1)).sum(dim=1) # (batch, hidden*2)
        else:
            # ── last hidden state of top layer, both directions ──
            # hidden shape: (num_layers * 2, batch, hidden_dim)
            forward_h  = hidden[-2]                               # (batch, hidden_dim)
            backward_h = hidden[-1]                               # (batch, hidden_dim)
            context    = torch.cat([forward_h, backward_h], dim=-1)  # (batch, hidden*2)

        context = self.dropout(context)
        logits  = self.classifier(context)    # (batch, num_classes)
        return logits


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
class WeightedCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    def forward(self, logits, targets):
        return self.ce(logits, targets)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
    }


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_loss_curves(train_losses, val_losses, title, path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses,   label="Val Loss",   marker="s")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_label_distribution(train_df, val_df, test_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (df, name) in zip(axes, [(train_df, "Train"), (val_df, "Validation"), (test_df, "Test")]):
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        bars   = ax.bar(LABELS, counts.values, color="#4e79a7")
        ax.set_title(f"{name} — Label Distribution")
        ax.tick_params(axis="x", rotation=45)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "label_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: label_distribution.png")


def save_all_runs_comparison(all_results):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    runs    = [r["run"] for r in all_results]
    colors  = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    for ax, metric in zip(axes, metrics):
        vals = [r["test"][metric] for r in all_results]
        bars = ax.bar(runs, vals, color=colors)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("BiLSTM — Test Set Comparison Across Runs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_runs_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: all_runs_comparison.png")


def save_per_class_f1_comparison(all_results, split="test"):
    """Bar chart of per-class F1 for each run — useful for error analysis in report."""
    run_names = [r["run"] for r in all_results]
    colors    = ["#4e79a7", "#f28e2b", "#e15759"]
    x         = np.arange(len(LABELS))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (result, color) in enumerate(zip(all_results, colors)):
        y_true       = result[f"{split}_labels"]
        y_pred       = result[f"{split}_preds"]
        per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        ax.bar(x + i * width, per_class_f1, width, label=result["run"], color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class F1 Comparison ({split.title()} Set) — BiLSTM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_class_f1_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: per_class_f1_comparison.png")


def save_summary_csv(all_results, split):
    rows = []
    for r in all_results:
        row = {"run": r["run"]}
        row.update(r[split])
        rows.append(row)
    out = os.path.join(OUT_DIR, f"{split}_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved: {split}_summary.csv")


# ─────────────────────────────────────────────
# TRAINING / EVAL LOOPS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        lengths   = batch["length"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss   = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        [ID2LABEL[i] for i in all_labels],
        [ID2LABEL[i] for i in all_preds],
    )
    return avg_loss, metrics, all_labels, all_preds


def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels, all_ids = 0.0, [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            lengths   = batch["length"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            logits = model(input_ids, lengths)
            loss   = loss_fn(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch["sample_id"])   # original CSV id column

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        [ID2LABEL[i] for i in all_labels],
        [ID2LABEL[i] for i in all_preds],
    )
    return avg_loss, metrics, all_labels, all_preds, all_ids


# ─────────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────
def run_experiment(run_name, train_df, val_df, test_df, vocab,
                   use_eda=False, use_attention=False):

    print(f"\n{'='*60}")
    print(f"  RUN      : {run_name}")
    print(f"  EDA      : {use_eda}  |  Attention : {use_attention}")
    print(f"{'='*60}\n")

    set_seed()

    # ── augment if needed ──
    train_data = augment_minority_classes(train_df.copy()) if use_eda else train_df.copy()
    print(f"  Train: {len(train_data)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # ── class weights (computed from training data) ──
    y_train      = train_data["label"].map(LABEL2ID).values
    cw           = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float)

    # ── datasets & loaders ──
    train_ds = RedditDataset(train_data, vocab)
    val_ds   = RedditDataset(val_df,     vocab)
    test_ds  = RedditDataset(test_df,    vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── model ──
    model = BiLSTMClassifier(
        vocab_size    = len(vocab),
        embed_dim     = EMBED_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
        num_classes   = NUM_LABELS,
        dropout       = DROPOUT,
        use_attention = use_attention,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── loss & optimizer ──
    loss_fn   = WeightedCELoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── training loop ──
    best_val_f1  = -1
    best_state   = None
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_met, _, _        = train_epoch(model, train_loader, optimizer, loss_fn)
        vl_loss, vl_met, _, _, _     = eval_epoch(model, val_loader,   loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"  Epoch {epoch:>2}/{EPOCHS} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_met['accuracy']:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_met['accuracy']:.4f} F1 {vl_met['f1_macro']:.4f}")

        if vl_met["f1_macro"] > best_val_f1:
            best_val_f1  = vl_met["f1_macro"]
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    # ── reload best checkpoint ──
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # ── evaluate on all three splits ──
    _, tr_met, tr_lbl, tr_prd, _      = eval_epoch(model, train_loader, loss_fn)
    _, vl_met, vl_lbl, vl_prd, _      = eval_epoch(model, val_loader,   loss_fn)
    _, ts_met, ts_lbl, ts_prd, ts_ids = eval_epoch(model, test_loader,  loss_fn)

    tr_lbl_str = [ID2LABEL[i] for i in tr_lbl]; tr_prd_str = [ID2LABEL[i] for i in tr_prd]
    vl_lbl_str = [ID2LABEL[i] for i in vl_lbl]; vl_prd_str = [ID2LABEL[i] for i in vl_prd]
    ts_lbl_str = [ID2LABEL[i] for i in ts_lbl]; ts_prd_str = [ID2LABEL[i] for i in ts_prd]

    print(f"\n  Train — Acc {tr_met['accuracy']:.4f}  F1 {tr_met['f1_macro']:.4f}")
    print(f"  Val   — Acc {vl_met['accuracy']:.4f}  F1 {vl_met['f1_macro']:.4f}")
    print(f"  Test  — Acc {ts_met['accuracy']:.4f}  F1 {ts_met['f1_macro']:.4f}")

    # ── save all outputs ──
    prefix = os.path.join(OUT_DIR, run_name)

    # loss curves
    save_loss_curves(train_losses, val_losses,
                     f"{run_name} — Loss Curves",
                     f"{prefix}_loss_curves.png")

    # confusion matrices (train, val, test)
    for split_name, y_t, y_p in [("train", tr_lbl_str, tr_prd_str),
                                  ("val",   vl_lbl_str, vl_prd_str),
                                  ("test",  ts_lbl_str, ts_prd_str)]:
        save_confusion_matrix(y_t, y_p,
                              f"{run_name} — {split_name.title()} Confusion Matrix",
                              f"{prefix}_{split_name}_confusion_matrix.png")

    # classification reports
    for split_name, y_t, y_p in [("val",  vl_lbl_str, vl_prd_str),
                                  ("test", ts_lbl_str, ts_prd_str)]:
        report = classification_report(y_t, y_p, labels=LABELS, zero_division=0)
        with open(f"{prefix}_{split_name}_classification_report.txt", "w") as f:
            f.write(f"Run: {run_name}\nSplit: {split_name}\n\n{report}")

    # metrics JSON
    with open(f"{prefix}_val_metrics.json",  "w") as f:
        json.dump({"run": run_name, **vl_met}, f, indent=2)
    with open(f"{prefix}_test_metrics.json", "w") as f:
        json.dump({"run": run_name, **ts_met}, f, indent=2)

    # misclassified examples (test)
    misclassified = [
        {"id": ts_ids[i], "text": test_df.iloc[i]["text"][:300],
         "true_label": ts_lbl_str[i], "pred_label": ts_prd_str[i]}
        for i in range(len(ts_lbl_str)) if ts_lbl_str[i] != ts_prd_str[i]
    ]
    pd.DataFrame(misclassified).to_csv(f"{prefix}_test_misclassified.csv", index=False)

    # ── CodaBench submission CSV (id, label) ──
    submission_df   = pd.DataFrame({"id": ts_ids, "label": ts_prd_str})
    submission_path = f"{prefix}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"  Saved: {run_name}_submission.csv  ({len(submission_df)} rows)")
    print(f"  All outputs saved to diagrams/{run_name}_*")

    return {
        "run":             run_name,
        "use_eda":         use_eda,
        "use_attention":   use_attention,
        "val":             vl_met,
        "test":            ts_met,
        "val_labels":      vl_lbl_str,
        "val_preds":       vl_prd_str,
        "test_labels":     ts_lbl_str,
        "test_preds":      ts_prd_str,
        "submission_path": submission_path,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── load data ──
    print("\nLoading data...")
    train_df = load_split("train.csv")
    val_df   = load_split("validation.csv")
    test_df  = load_split("test.csv")

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())

    save_label_distribution(train_df, val_df, test_df)

    # ── build vocabulary from training data only ──
    print("\nBuilding vocabulary from training data...")
    vocab = build_vocab(train_df["text"].tolist(), min_freq=VOCAB_MIN_FREQ)
    print(f"  Vocabulary size: {len(vocab):,} tokens  (min_freq={VOCAB_MIN_FREQ})")

    # ── Run 1: BiLSTM base — no augmentation, no attention ────────────────
    r1 = run_experiment(
        run_name      = "run1_bilstm_base",
        train_df      = train_df,
        val_df        = val_df,
        test_df       = test_df,
        vocab         = vocab,
        use_eda       = False,
        use_attention = False,
    )

    # ── Run 2: BiLSTM + EDA augmentation ──────────────────────────────────
    r2 = run_experiment(
        run_name      = "run2_bilstm_eda",
        train_df      = train_df,
        val_df        = val_df,
        test_df       = test_df,
        vocab         = vocab,
        use_eda       = True,
        use_attention = False,
    )

    # ── Run 3: BiLSTM + Attention ──────────────────────────────────────────
    r3 = run_experiment(
        run_name      = "run3_bilstm_attention",
        train_df      = train_df,
        val_df        = val_df,
        test_df       = test_df,
        vocab         = vocab,
        use_eda       = False,
        use_attention = True,
    )

    all_results = [r1, r2, r3]

    # ── comparison plots ──
    print("\nGenerating comparison plots...")
    save_all_runs_comparison(all_results)
    save_per_class_f1_comparison(all_results, split="test")
    save_summary_csv(all_results, "val")
    save_summary_csv(all_results, "test")

    # ── pick best run by val F1 and create final submission.zip ──
    best = max(all_results, key=lambda r: r["val"]["f1_macro"])
    print(f"\nBest run by val F1: {best['run']}  (val F1 = {best['val']['f1_macro']:.4f})")

    zip_path = os.path.join(OUT_DIR, "best_submission.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(best["submission_path"], arcname="submission.csv")
    print(f"  Saved: best_submission.zip  (based on {best['run']})")

    # ── final summary table ──
    print("\n" + "="*65)
    print("  FINAL RESULTS SUMMARY")
    print("="*65)
    print(f"{'Run':<35} | {'Acc':>6} | {'F1 Mac':>7} | {'F1 Wt':>7}")
    print(f"{'-'*35}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
    for r in all_results:
        t = r["test"]
        print(f"{r['run']:<35} | {t['accuracy']:>6.4f} | {t['f1_macro']:>7.4f} | {t['f1_weighted']:>7.4f}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Submit to CodaBench:  {zip_path}")
