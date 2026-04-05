"""
Rayan Nasrallah — COMPSCI 4NL3 Final Project
Model: BiLSTM classifier on top of frozen Flair pretrained character BiLSTM embeddings
Task: 5-class Reddit comment classification
      (ADVICE, ANECDOTE, APPRAISAL, EMOTIONAL_SUPPORT, WARNING)

Architecture:
  Flair 'news-forward-fast' is a character-level Bidirectional LSTM language model
  pretrained on a large English news corpus (NOT a transformer). It produces
  1024-dim contextual token embeddings where each word's representation depends
  on all surrounding characters and words in the full sentence.

  These Flair BiLSTM embeddings are frozen (not updated). A small task-specific
  BiLSTM classifier is trained on top:
    Flair character BiLSTM (frozen, pretrained, 1024-dim output per token)
      -> Task BiLSTM (hidden=256, bidirectional -> 512-dim)
      -> Dropout -> Linear(512, 5)

3 Ablation Runs:
  Run 1: Flair features + BiLSTM classifier + Weighted CE         (base)
  Run 2: Flair features + BiLSTM classifier + EDA + Weighted CE  (+ augmentation)
  Run 3: Flair features + BiLSTM + Attention + Weighted CE       (+ attention)

Outputs (all saved to rayan_model/diagrams/):
  - label_distribution.png
  - runN_loss_curves.png
  - runN_train/val/test_confusion_matrix.png
  - runN_val/test_classification_report.txt
  - runN_val/test_metrics.json
  - runN_test_misclassified.csv
  - runN_submission.csv          <-- CodaBench format (id, label)
  - all_runs_comparison.png
  - per_class_f1_comparison.png
  - val_summary.csv / test_summary.csv
  - best_submission.zip          <-- final CodaBench submission

Setup (run once on the VM):
  pip install torch scikit-learn matplotlib seaborn nltk pandas numpy flair
  python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

  Flair model (~74MB) downloads automatically on first run.

Run:
  cd final_step/
  python rayan_model/train.py
"""

import os
import json
import random
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
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

from flair.embeddings import FlairEmbeddings
from flair.data import Sentence

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED          = 42
MAX_LEN       = 64        # tokens per sequence (Flair tokenizes internally)
FLAIR_DIM     = 1024      # news-forward-fast output dimension
HIDDEN_DIM    = 256       # task BiLSTM hidden per direction (512 after concat)
NUM_LAYERS    = 1
DROPOUT       = 0.5
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5
EXTRACT_BATCH = 64        # sentences per Flair forward pass
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(f"Model  : Flair BiLSTM (frozen, {FLAIR_DIM}d) -> task BiLSTM (hidden={HIDDEN_DIM})")


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
    non_stop = [w for w in words if w.lower() not in STOP_WORDS]
    random.shuffle(non_stop)
    replaced = 0
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
    words = sentence.split()
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
# FLAIR EMBEDDING EXTRACTION
# ─────────────────────────────────────────────
def load_flair_model():
    """
    Load Flair 'news-forward-fast': a character-level BiLSTM language model
    pretrained on English news text. ~74MB, cached after first download.
    Output: 1024-dim contextual embedding per token.
    """
    print("  Loading Flair 'news-forward-fast' pretrained BiLSTM (~74MB first run)...")
    model = FlairEmbeddings("news-forward-fast")
    print("  Flair model ready.")
    return model


def extract_embeddings(texts, flair_model, desc=""):
    """
    Pass texts through the frozen Flair BiLSTM to get contextual token embeddings.
    Returns:
      embeddings : (N, MAX_LEN, FLAIR_DIM) float32 CPU tensor
      lengths    : (N,) int64 CPU tensor
    """
    all_embs = []
    all_lens = []
    n = len(texts)

    for i in range(0, n, EXTRACT_BATCH):
        batch_texts = texts[i: i + EXTRACT_BATCH]
        sentences   = [Sentence(t if t.strip() else "empty") for t in batch_texts]

        # Truncate long sentences before embedding to save time and memory
        for sent in sentences:
            if len(sent) > MAX_LEN:
                sent.tokens = sent.tokens[:MAX_LEN]

        flair_model.embed(sentences)

        for sent in sentences:
            tok_embs = torch.stack([tok.embedding.detach().cpu() for tok in sent])
            n_tok    = tok_embs.shape[0]
            length   = max(min(n_tok, MAX_LEN), 1)

            if n_tok < MAX_LEN:
                pad      = torch.zeros(MAX_LEN - n_tok, FLAIR_DIM)
                tok_embs = torch.cat([tok_embs, pad], dim=0)
            else:
                tok_embs = tok_embs[:MAX_LEN]

            all_embs.append(tok_embs)
            all_lens.append(length)

            # Free GPU/CPU memory held by Flair token embeddings
            for tok in sent:
                tok.clear_embeddings()

        done = min(i + EXTRACT_BATCH, n)
        if done % 128 == 0 or done == n:
            print(f"    [{desc}] {done}/{n} texts...")

    return torch.stack(all_embs), torch.tensor(all_lens, dtype=torch.long)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    """Dataset backed by pre-computed Flair embeddings."""

    def __init__(self, embeddings, lengths, df):
        self.embeddings = embeddings      # (N, MAX_LEN, FLAIR_DIM)
        self.lengths    = lengths         # (N,)
        self.labels     = [LABEL2ID[l] for l in df["label"].tolist()]
        self.ids        = df["id"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "length":     self.lengths[idx],
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_id":  self.ids[idx],
        }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Task BiLSTM trained on top of frozen Flair contextual embeddings.

    Without attention (Runs 1 & 2):
      Concatenate the final forward and backward hidden states of the LSTM
      and feed to the linear classifier.

    With attention (Run 3):
      A learned linear projection scores every timestep. Softmax turns scores
      into weights. The classification vector is the weighted sum of all hidden
      states, letting the model focus on the most informative words.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 dropout, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attn_proj = nn.Linear(hidden_dim * 2, 1)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings, lengths):
        x = self.dropout(embeddings)   # (B, T, FLAIR_DIM)

        packed             = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(packed)
        output, _          = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True
        )  # (B, T, hidden*2)

        if self.use_attention:
            scores  = self.attn_proj(output).squeeze(-1)         # (B, T)
            max_t   = output.shape[1]
            pad_mask = (torch.arange(max_t, device=lengths.device)
                        .unsqueeze(0) >= lengths.unsqueeze(1))   # True = pad
            scores  = scores.masked_fill(pad_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            context = (output * weights.unsqueeze(-1)).sum(dim=1) # (B, hidden*2)
        else:
            forward_h  = hidden[-2]
            backward_h = hidden[-1]
            context    = torch.cat([forward_h, backward_h], dim=-1)

        return self.classifier(self.dropout(context))


# ─────────────────────────────────────────────
# LOSS
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
    plt.suptitle("Flair BiLSTM — Test Set Comparison Across Runs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_runs_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: all_runs_comparison.png")


def save_per_class_f1_comparison(all_results, split="test"):
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    x      = np.arange(len(LABELS))
    width  = 0.25

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
    ax.set_title(f"Per-Class F1 Comparison ({split.title()} Set) — Flair BiLSTM")
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
        embs   = batch["embeddings"].to(DEVICE)
        lens   = batch["length"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(embs, lens)
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
    return avg_loss, metrics


def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels, all_ids = 0.0, [], [], []

    with torch.no_grad():
        for batch in loader:
            embs   = batch["embeddings"].to(DEVICE)
            lens   = batch["length"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(embs, lens)
            loss   = loss_fn(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch["sample_id"])

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        [ID2LABEL[i] for i in all_labels],
        [ID2LABEL[i] for i in all_preds],
    )
    return avg_loss, metrics, all_labels, all_preds, all_ids


# ─────────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────
def run_experiment(run_name, train_df, val_df, test_df,
                   val_embs, val_lens, test_embs, test_lens,
                   flair_model, use_eda=False, use_attention=False):

    print(f"\n{'='*60}")
    print(f"  RUN      : {run_name}")
    print(f"  EDA      : {use_eda}  |  Attention : {use_attention}")
    print(f"{'='*60}\n")

    set_seed()

    # augment text first, then extract Flair embeddings for the (possibly larger) train set
    train_data = augment_minority_classes(train_df.copy()) if use_eda else train_df.copy()
    print(f"  Train: {len(train_data)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    print("  Extracting Flair embeddings for training set...")
    train_embs, train_lens = extract_embeddings(
        train_data["text"].tolist(), flair_model, desc="Train"
    )

    train_ds = EmbeddingDataset(train_embs, train_lens, train_data)
    val_ds   = EmbeddingDataset(val_embs,   val_lens,   val_df)
    test_ds  = EmbeddingDataset(test_embs,  test_lens,  test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_train      = train_data["label"].map(LABEL2ID).values
    cw           = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float)

    model = BiLSTMClassifier(
        input_dim     = FLAIR_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
        num_classes   = NUM_LABELS,
        dropout       = DROPOUT,
        use_attention = use_attention,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Task BiLSTM trainable parameters: {total_params:,}")

    loss_fn   = WeightedCELoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1  = -1
    best_state   = None
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_met          = train_epoch(model, train_loader, optimizer, loss_fn)
        vl_loss, vl_met, _, _, _ = eval_epoch(model, val_loader,   loss_fn)

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

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    _, tr_met, tr_lbl, tr_prd, _      = eval_epoch(model, train_loader, loss_fn)
    _, vl_met, vl_lbl, vl_prd, _      = eval_epoch(model, val_loader,   loss_fn)
    _, ts_met, ts_lbl, ts_prd, ts_ids = eval_epoch(model, test_loader,  loss_fn)

    tr_lbl_str = [ID2LABEL[i] for i in tr_lbl]; tr_prd_str = [ID2LABEL[i] for i in tr_prd]
    vl_lbl_str = [ID2LABEL[i] for i in vl_lbl]; vl_prd_str = [ID2LABEL[i] for i in vl_prd]
    ts_lbl_str = [ID2LABEL[i] for i in ts_lbl]; ts_prd_str = [ID2LABEL[i] for i in ts_prd]

    print(f"\n  Train -- Acc {tr_met['accuracy']:.4f}  F1 {tr_met['f1_macro']:.4f}")
    print(f"  Val   -- Acc {vl_met['accuracy']:.4f}  F1 {vl_met['f1_macro']:.4f}")
    print(f"  Test  -- Acc {ts_met['accuracy']:.4f}  F1 {ts_met['f1_macro']:.4f}")

    prefix = os.path.join(OUT_DIR, run_name)

    save_loss_curves(train_losses, val_losses,
                     f"{run_name} Loss Curves", f"{prefix}_loss_curves.png")

    for split_name, y_t, y_p in [("train", tr_lbl_str, tr_prd_str),
                                  ("val",   vl_lbl_str, vl_prd_str),
                                  ("test",  ts_lbl_str, ts_prd_str)]:
        save_confusion_matrix(y_t, y_p,
                              f"{run_name} {split_name.title()} Confusion Matrix",
                              f"{prefix}_{split_name}_confusion_matrix.png")

    for split_name, y_t, y_p in [("val",  vl_lbl_str, vl_prd_str),
                                  ("test", ts_lbl_str, ts_prd_str)]:
        report = classification_report(y_t, y_p, labels=LABELS, zero_division=0)
        with open(f"{prefix}_{split_name}_classification_report.txt", "w") as f:
            f.write(f"Run: {run_name}\nSplit: {split_name}\n\n{report}")

    with open(f"{prefix}_val_metrics.json",  "w") as f:
        json.dump({"run": run_name, **vl_met}, f, indent=2)
    with open(f"{prefix}_test_metrics.json", "w") as f:
        json.dump({"run": run_name, **ts_met}, f, indent=2)

    misclassified = [
        {"id": ts_ids[i], "text": test_df.iloc[i]["text"][:300],
         "true_label": ts_lbl_str[i], "pred_label": ts_prd_str[i]}
        for i in range(len(ts_lbl_str)) if ts_lbl_str[i] != ts_prd_str[i]
    ]
    pd.DataFrame(misclassified).to_csv(f"{prefix}_test_misclassified.csv", index=False)

    submission_df   = pd.DataFrame({"id": ts_ids, "label": ts_prd_str})
    submission_path = f"{prefix}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"  Saved: {run_name}_submission.csv  ({len(submission_df)} rows)")

    del train_embs, train_lens, train_ds   # free RAM before next run

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

    print("\nLoading data...")
    train_df = load_split("train.csv")
    val_df   = load_split("validation.csv")
    test_df  = load_split("test.csv")

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())

    save_label_distribution(train_df, val_df, test_df)

    # Load Flair pretrained BiLSTM once — shared across all 3 runs
    print("\nLoading Flair pretrained BiLSTM language model...")
    flair_model = load_flair_model()

    # Pre-extract val and test embeddings once — reused by all 3 runs
    print("\nExtracting Flair embeddings for val and test sets (done once)...")
    val_embs,  val_lens  = extract_embeddings(val_df["text"].tolist(),  flair_model, "Val")
    test_embs, test_lens = extract_embeddings(test_df["text"].tolist(), flair_model, "Test")

    # Run 1 — base: Flair BiLSTM, no EDA, no attention
    r1 = run_experiment(
        run_name="run1_bilstm_base",
        train_df=train_df, val_df=val_df, test_df=test_df,
        val_embs=val_embs, val_lens=val_lens,
        test_embs=test_embs, test_lens=test_lens,
        flair_model=flair_model,
        use_eda=False, use_attention=False,
    )

    # Run 2 — EDA: augment minority classes then extract embeddings
    r2 = run_experiment(
        run_name="run2_bilstm_eda",
        train_df=train_df, val_df=val_df, test_df=test_df,
        val_embs=val_embs, val_lens=val_lens,
        test_embs=test_embs, test_lens=test_lens,
        flair_model=flair_model,
        use_eda=True, use_attention=False,
    )

    # Run 3 — attention: soft attention over all hidden states
    r3 = run_experiment(
        run_name="run3_bilstm_attention",
        train_df=train_df, val_df=val_df, test_df=test_df,
        val_embs=val_embs, val_lens=val_lens,
        test_embs=test_embs, test_lens=test_lens,
        flair_model=flair_model,
        use_eda=False, use_attention=True,
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
