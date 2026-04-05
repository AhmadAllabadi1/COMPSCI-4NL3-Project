"""
Rayan Nasrallah -- COMPSCI 4NL3 Final Project
Model  : BiLSTM with frozen Flair pretrained character-level contextual embeddings
Task   : 5-class Reddit comment classification
         (ADVICE, ANECDOTE, APPRAISAL, EMOTIONAL_SUPPORT, WARNING)

Improved version:
- Uses packed sequences so padding does not corrupt the BiLSTM state
- Uses true final hidden state from the last BiLSTM layer
- Uses masked max/mean pooling (ignores padded timesteps)
- Pre-computes validation/test embeddings once and reuses them across runs
- Pre-computes train embeddings once per distinct train set (base / EDA)
- Restores the best checkpoint before final prediction
"""

import os
import json
import random
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import wordnet, stopwords

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED        = 42
BS          = 16         # training/inference batch size
EMBED_BS    = 32         # batch size for Flair embedding pass
DROPOUT     = 0.4
EPOCHS      = 15
LR          = 5e-4       # slightly safer than 1e-3 for this setup
PATIENCE    = 4          # early-stopping patience (val macro F1)
MAX_TOKENS  = 512        # cap length before Flair embedding
EMBEDDING_DIM = 2048     # news-forward-fast (1024) + news-backward-fast (1024)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS     = ["ADVICE", "ANECDOTE", "APPRAISAL", "EMOTIONAL_SUPPORT", "WARNING"]
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}
ID2LABEL   = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

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
print("Model  : BiLSTM + Flair frozen embeddings (packed + masked)")


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
    kept = [w for w in words if random.random() > p]
    return kept if kept else [random.choice(words)]


def eda(sentence, alpha=0.1, num_aug=1):
    words = sentence.split()
    if not words:
        return [sentence]
    n = max(1, int(alpha * len(words)))
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
        return pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_split(filename):
    path = os.path.join(BASE_DIR, filename)
    df   = pd.read_csv(path)
    df   = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df   = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# FLAIR EMBEDDING (pre-compute + cache)
# ─────────────────────────────────────────────
def build_flair_embeddings():
    print("  Loading Flair embeddings (downloads on first run ~200MB)...")
    fwd = FlairEmbeddings("news-forward-fast")
    bwd = FlairEmbeddings("news-backward-fast")
    return StackedEmbeddings([fwd, bwd])


def _text_key(texts):
    return hash(tuple(str(t).strip() for t in texts))


def embed_texts(texts, stacked_emb, cache=None, desc="Embedding"):
    """
    Pre-compute Flair embeddings for a list of raw text strings.
    Returns: list of tensors, each shape (seq_len, EMBEDDING_DIM)
    """
    key = _text_key(texts)
    if cache is not None and key in cache:
        print(f"    {desc}: using cached embeddings")
        return cache[key]

    all_tensors = []
    n = len(texts)
    for start in range(0, n, EMBED_BS):
        batch = texts[start:start + EMBED_BS]
        sentences = []
        for t in batch:
            t = str(t).strip() or "empty"
            sentences.append(Sentence(t[:MAX_TOKENS]))

        stacked_emb.embed(sentences)

        for sent in sentences:
            if len(sent.tokens) == 0:
                emb = torch.zeros(1, EMBEDDING_DIM)
            else:
                emb = torch.stack([tok.embedding.detach().cpu() for tok in sent.tokens])
            all_tensors.append(emb)
            sent.clear_embeddings()

        if ((start // EMBED_BS) + 1) % 5 == 0 or (start + EMBED_BS) >= n:
            print(f"    {desc}: {min(start + EMBED_BS, n)}/{n}")

    if cache is not None:
        cache[key] = all_tensors
    return all_tensors


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels     = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.embeddings[idx]
        label = self.labels[idx]
        length = seq.size(0)
        return seq, length, label


def collate_fn(batch):
    seqs, lengths, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    BiLSTM with packed sequences + masked concat pooling.
    Input: pre-computed Flair embeddings, shape (B, T, 2048)
    """
    def __init__(self, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        pool_dim = hidden_size * 2 * 3  # last + max + mean
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # True final hidden state from the last BiLSTM layer
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        last = torch.cat([last_forward, last_backward], dim=1)

        _, max_len, _ = out.size()
        mask = torch.arange(max_len, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        # Masked mean pooling
        summed = (out * mask_f).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        mean_p = summed / denom

        # Masked max pooling
        masked_out = out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_p = masked_out.max(dim=1).values
        max_p = torch.where(torch.isinf(max_p), torch.zeros_like(max_p), max_p)

        pooled = torch.cat([last, max_p, mean_p], dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
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
    n = min(len(train_losses), len(val_losses))
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses[:n], label="Train Loss", marker="o")
    plt.plot(val_losses[:n], label="Val Loss", marker="s")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_label_distribution(train_df, val_df, test_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (df, name) in zip(axes, [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]):
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        bars = ax.bar(LABELS, counts.values, color="#4e79a7")
        ax.set_title(f"{name} Label Distribution")
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
    runs = [r["run"] for r in all_results]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
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
    plt.suptitle("Flair BiLSTM -- Test Set Comparison Across Runs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_runs_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: all_runs_comparison.png")


def save_per_class_f1_comparison(all_results, split="test"):
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    x = np.arange(len(LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (result, color) in enumerate(zip(all_results, colors)):
        y_true = result[f"{split}_labels"]
        y_pred = result[f"{split}_preds"]
        per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        ax.bar(x + i * width, per_class_f1, width, label=result["run"], color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class F1 Comparison ({split.title()} Set) -- Flair BiLSTM")
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
# PREDICTION HELPER
# ─────────────────────────────────────────────
def get_predictions(model, embeddings):
    model.eval()
    dataset = EmbeddingDataset(embeddings, [0] * len(embeddings))
    loader = DataLoader(dataset, batch_size=BS, shuffle=False, collate_fn=collate_fn)
    all_probs = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(x, lengths)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    preds = [ID2LABEL[i] for i in all_probs.argmax(axis=1)]
    return all_probs, preds


# ─────────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────
def run_experiment(run_name, train_df, val_df, test_df,
                   tr_emb, vl_emb, ts_emb,
                   use_eda=False, num_layers=2, hidden_size=512):
    print(f"\n{'='*60}")
    print(f"  RUN        : {run_name}")
    print(f"  EDA        : {use_eda}")
    print(f"  LSTM Layers: {num_layers}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"{'='*60}\n")

    set_seed()

    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    y_train = train_df["label"].map(LABEL2ID).values
    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y_train)
    weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    tr_labels = train_df["label"].map(LABEL2ID).tolist()
    vl_labels = val_df["label"].map(LABEL2ID).tolist()
    ts_labels = test_df["label"].map(LABEL2ID).tolist()

    tr_loader = DataLoader(EmbeddingDataset(tr_emb, tr_labels), batch_size=BS, shuffle=True, collate_fn=collate_fn)
    vl_loader = DataLoader(EmbeddingDataset(vl_emb, vl_labels), batch_size=BS, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMClassifier(
        embedding_dim=EMBEDDING_DIM,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=NUM_LABELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    best_val_f1 = -1.0
    best_state = None
    patience_count = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x, lengths, y in tr_loader:
            x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(tr_loader))

        model.eval()
        val_loss = 0.0
        vl_preds = []
        vl_true = []
        with torch.no_grad():
            for x, lengths, y in vl_loader:
                x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                logits = model(x, lengths)
                val_loss += criterion(logits, y).item()
                vl_preds += logits.argmax(dim=-1).cpu().tolist()
                vl_true += y.cpu().tolist()
        val_losses.append(val_loss / len(vl_loader))

        vl_f1 = f1_score(vl_true, vl_preds, average="macro", zero_division=0)
        scheduler.step(vl_f1)

        if epoch % 2 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/{EPOCHS}  train_loss={train_losses[-1]:.4f}  "
                f"val_loss={val_losses[-1]:.4f}  val_f1={vl_f1:.4f}"
            )

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)

    _, tr_prd_str = get_predictions(model, tr_emb)
    _, vl_prd_str = get_predictions(model, vl_emb)
    _, ts_prd_str = get_predictions(model, ts_emb)

    tr_lbl_str = train_df["label"].tolist()
    vl_lbl_str = val_df["label"].tolist()
    ts_lbl_str = test_df["label"].tolist()
    ts_ids = test_df["id"].tolist()

    tr_met = compute_metrics(tr_lbl_str, tr_prd_str)
    vl_met = compute_metrics(vl_lbl_str, vl_prd_str)
    ts_met = compute_metrics(ts_lbl_str, ts_prd_str)

    print(f"\n  Train -- Acc {tr_met['accuracy']:.4f}  F1 {tr_met['f1_macro']:.4f}")
    print(f"  Val   -- Acc {vl_met['accuracy']:.4f}  F1 {vl_met['f1_macro']:.4f}")
    print(f"  Test  -- Acc {ts_met['accuracy']:.4f}  F1 {ts_met['f1_macro']:.4f}")

    prefix = os.path.join(OUT_DIR, run_name)

    save_loss_curves(train_losses, val_losses, f"{run_name} Loss Curves", f"{prefix}_loss_curves.png")

    for split_name, y_t, y_p in [
        ("train", tr_lbl_str, tr_prd_str),
        ("val", vl_lbl_str, vl_prd_str),
        ("test", ts_lbl_str, ts_prd_str),
    ]:
        save_confusion_matrix(
            y_t, y_p,
            f"{run_name} {split_name.title()} Confusion Matrix",
            f"{prefix}_{split_name}_confusion_matrix.png",
        )

    for split_name, y_t, y_p in [("val", vl_lbl_str, vl_prd_str), ("test", ts_lbl_str, ts_prd_str)]:
        report = classification_report(y_t, y_p, labels=LABELS, zero_division=0)
        with open(f"{prefix}_{split_name}_classification_report.txt", "w") as f:
            f.write(f"Run: {run_name}\nSplit: {split_name}\n\n{report}")

    with open(f"{prefix}_val_metrics.json", "w") as f:
        json.dump({"run": run_name, **vl_met}, f, indent=2)
    with open(f"{prefix}_test_metrics.json", "w") as f:
        json.dump({"run": run_name, **ts_met}, f, indent=2)

    misclassified = [
        {
            "id": ts_ids[i],
            "text": test_df.iloc[i]["text"][:300],
            "true_label": ts_lbl_str[i],
            "pred_label": ts_prd_str[i],
        }
        for i in range(len(ts_lbl_str)) if ts_lbl_str[i] != ts_prd_str[i]
    ]
    pd.DataFrame(misclassified).to_csv(f"{prefix}_test_misclassified.csv", index=False)

    submission_df = pd.DataFrame({"id": ts_ids, "label": ts_prd_str})
    submission_path = f"{prefix}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"  Saved: {run_name}_submission.csv  ({len(submission_df)} rows)")
    print(f"  All outputs saved to diagrams/{run_name}_*")

    return {
        "run": run_name,
        "use_eda": use_eda,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "val": vl_met,
        "test": ts_met,
        "val_labels": vl_lbl_str,
        "val_preds": vl_prd_str,
        "test_labels": ts_lbl_str,
        "test_preds": ts_prd_str,
        "submission_path": submission_path,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading data...")
    train_df = load_split("train.csv")
    val_df   = load_split("validation.csv")
    test_df  = load_split("test.csv")

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())

    save_label_distribution(train_df, val_df, test_df)

    stacked_emb = build_flair_embeddings()
    emb_cache = {}

    # Precompute shared validation/test embeddings once.
    print("\nPrecomputing shared validation/test embeddings...")
    vl_emb = embed_texts(val_df["text"].tolist(), stacked_emb, cache=emb_cache, desc="Val")
    ts_emb = embed_texts(test_df["text"].tolist(), stacked_emb, cache=emb_cache, desc="Test")

    # Prepare distinct training variants once.
    train_base_df = train_df.copy().reset_index(drop=True)
    train_eda_df  = augment_minority_classes(train_df.copy())

    print("\nPrecomputing training embeddings...")
    tr_base_emb = embed_texts(train_base_df["text"].tolist(), stacked_emb, cache=emb_cache, desc="Train Base")
    tr_eda_emb  = embed_texts(train_eda_df["text"].tolist(), stacked_emb, cache=emb_cache, desc="Train EDA")

    r1 = run_experiment(
        run_name="run1_flair_bilstm_base",
        train_df=train_base_df,
        val_df=val_df,
        test_df=test_df,
        tr_emb=tr_base_emb,
        vl_emb=vl_emb,
        ts_emb=ts_emb,
        use_eda=False,
        num_layers=2,
        hidden_size=512,
    )

    r2 = run_experiment(
        run_name="run2_flair_bilstm_eda",
        train_df=train_eda_df,
        val_df=val_df,
        test_df=test_df,
        tr_emb=tr_eda_emb,
        vl_emb=vl_emb,
        ts_emb=ts_emb,
        use_eda=True,
        num_layers=2,
        hidden_size=512,
    )

    r3 = run_experiment(
        run_name="run3_flair_bilstm_deep_eda",
        train_df=train_eda_df,
        val_df=val_df,
        test_df=test_df,
        tr_emb=tr_eda_emb,
        vl_emb=vl_emb,
        ts_emb=ts_emb,
        use_eda=True,
        num_layers=3,
        hidden_size=512,
    )

    all_results = [r1, r2, r3]

    print("\nGenerating comparison plots...")
    save_all_runs_comparison(all_results)
    save_per_class_f1_comparison(all_results, split="test")
    save_summary_csv(all_results, "val")
    save_summary_csv(all_results, "test")

    best = max(all_results, key=lambda r: r["val"]["f1_macro"])
    print(f"\nBest run by val F1: {best['run']}  (val F1 = {best['val']['f1_macro']:.4f})")

    zip_path = os.path.join(OUT_DIR, "best_submission.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(best["submission_path"], arcname="submission.csv")
    print(f"  Saved: best_submission.zip  (based on {best['run']})")

    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Run':<42} | {'Acc':>6} | {'F1 Mac':>7} | {'F1 Wt':>7}")
    print(f"{'-' * 42}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}")
    for r in all_results:
        t = r["test"]
        print(f"{r['run']:<42} | {t['accuracy']:>6.4f} | {t['f1_macro']:>7.4f} | {t['f1_weighted']:>7.4f}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Submit to CodaBench:  {zip_path}")