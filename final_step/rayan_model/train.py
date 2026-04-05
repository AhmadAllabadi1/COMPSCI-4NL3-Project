"""
Rayan Nasrallah -- COMPSCI 4NL3 Final Project
Model  : ULMFiT (AWD-LSTM) pretrained language model, fine-tuned for classification
Task   : 5-class Reddit comment classification
         (ADVICE, ANECDOTE, APPRAISAL, EMOTIONAL_SUPPORT, WARNING)

TRUE PRETRAINED LSTM ENCODER -- not a transformer, not just static embeddings.

Architecture:
  AWD-LSTM (Merity et al., 2017) -- 3-layer Bidirectional LSTM with weight
  dropping, variational dropout, and ASGD-based optimization. Pretrained on
  WikiText-103 (~103M tokens) via the fastai ULMFiT framework (Howard & Ruder,
  2018). The encoder captures broad English language structure before seeing any
  task data.

  Classification head: concat-pooling (last hidden + max-pool + mean-pool over
  all timesteps) -> BatchNorm -> Dropout -> Linear -> ReLU -> BatchNorm ->
  Dropout -> Linear(5).

  Fine-tuning strategy: gradual unfreezing + discriminative learning rates
  (ULMFiT approach designed specifically for small labeled datasets).

3 Ablation Runs:
  Run 1: Frozen AWD-LSTM encoder, only classifier head trained        (base)
  Run 2: Frozen encoder + EDA augmentation on minority classes        (+ data)
  Run 3: Frozen -> gradual full unfreeze + EDA + discriminative LRs  (+ unfreeze)

  Run 1 vs 2: measures the effect of augmentation with a frozen encoder.
  Run 2 vs 3: measures how much fine-tuning the encoder itself adds beyond
              just adapting the classification head.

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
  pip install fastai
  python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

  AWD-LSTM pretrained weights (~24MB) download automatically on first run from
  fast.ai model zoo.

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
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- must come before pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import wordnet, stopwords

from fastai.text.all import (
    TextDataLoaders,
    text_classifier_learner,
    AWD_LSTM,
    CrossEntropyLossFlat,
    Callback,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED    = 42
BS      = 16        # small batch -- better gradient signal on 1400 examples
DROP    = 0.3       # drop_mult: scales all AWD-LSTM dropout rates down

# Frozen-only training (Runs 1 & 2)
EPOCHS_FROZEN = 6
LR_HEAD       = 2e-2   # one-cycle LR for classifier head

# Gradual-unfreeze phase (Run 3 only -- stacks on top of frozen phase)
EPOCHS_UNFREEZE = 8
LR_UNFREEZE     = slice(1e-5, 1e-3)   # discriminative LR: low for early layers, high for last

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
print(f"Model  : ULMFiT AWD-LSTM (pretrained, WikiText-103)")


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
# LOSS TRACKER CALLBACK
# ─────────────────────────────────────────────
class EpochLossTracker(Callback):
    """Records mean training loss per epoch for loss-curve plots."""

    def before_fit(self):
        self.epoch_train_losses = []
        self._batch_buf         = []

    def after_batch(self):
        if self.training:
            self._batch_buf.append(float(self.smooth_loss))

    def after_epoch(self):
        if self._batch_buf:
            self.epoch_train_losses.append(float(np.mean(self._batch_buf)))
            self._batch_buf = []


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
    for ax, (df, name) in zip(axes, [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]):
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        bars   = ax.bar(LABELS, counts.values, color="#4e79a7")
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
    plt.suptitle("ULMFiT AWD-LSTM -- Test Set Comparison Across Runs", fontsize=13, y=1.02)
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
    ax.set_title(f"Per-Class F1 Comparison ({split.title()} Set) -- ULMFiT AWD-LSTM")
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
def get_predictions(learn, texts):
    """
    Run inference on a list of raw text strings.
    Returns: (probs tensor, pred_label_strings list)
    Predictions are returned in the same order as the input list.
    """
    dl    = learn.dls.test_dl(texts, bs=BS, shuffle_fn=None)
    probs = learn.get_preds(dl=dl, reorder=False, act=torch.nn.Softmax(dim=-1))[0]
    vocab = list(learn.dls.vocab[1])       # dls.vocab = (text_vocab, label_vocab)
    preds = [vocab[i] for i in probs.argmax(dim=-1).tolist()]
    return probs, preds


# ─────────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────
def run_experiment(run_name, train_df, val_df, test_df,
                   use_eda=False, gradual_unfreeze=False):
    """
    use_eda           : augment minority training classes with EDA before training.
    gradual_unfreeze  : after the frozen phase, unfreeze all layers and fine-tune
                        with discriminative learning rates (Run 3 only).
    """
    print(f"\n{'='*60}")
    print(f"  RUN              : {run_name}")
    print(f"  EDA              : {use_eda}")
    print(f"  Gradual Unfreeze : {gradual_unfreeze}")
    print(f"{'='*60}\n")

    set_seed()

    train_data = augment_minority_classes(train_df.copy()) if use_eda else train_df.copy()
    print(f"  Train: {len(train_data)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # ── Build fastai TextDataLoaders ──────────────────────────────────────
    # Combine train + val with an is_valid column so fastai knows the split.
    combined = pd.concat([
        train_data[["text", "label"]].assign(is_valid=False),
        val_df[["text", "label"]].assign(is_valid=True),
    ], ignore_index=True)

    dls = TextDataLoaders.from_df(
        combined,
        text_col  = "text",
        label_col = "label",
        valid_col = "is_valid",
        bs        = BS,
        seed      = SEED,
    )

    fastai_vocab = list(dls.vocab[1])   # dls.vocab = (text_vocab, label_vocab)
    print(f"  fastai label vocab : {fastai_vocab}")

    # ── Class weights -- mapped to fastai's vocab order ───────────────────
    y_train = train_data["label"].map(LABEL2ID).values
    cw      = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y_train)
    # Build weight tensor aligned with fastai's vocab (not our LABEL2ID order)
    wt = torch.zeros(NUM_LABELS, dtype=torch.float)
    for i, lbl in enumerate(fastai_vocab):
        if lbl in LABEL2ID:
            wt[i] = float(cw[LABEL2ID[lbl]])
    wt = wt.to(DEVICE)

    # ── Create ULMFiT classifier ──────────────────────────────────────────
    # pretrained=True downloads AWD-LSTM weights pretrained on WikiText-103.
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult  = DROP,
        pretrained = True,
        metrics    = [],
    )
    learn.loss_func = CrossEntropyLossFlat(weight=wt)

    # ── Training ──────────────────────────────────────────────────────────
    tracker      = EpochLossTracker()
    train_losses = []
    val_losses   = []

    # Phase 1: frozen encoder -- train only the classification head
    learn.freeze()
    learn.fit_one_cycle(EPOCHS_FROZEN, LR_HEAD, cbs=[tracker])
    train_losses += tracker.epoch_train_losses.copy()
    val_losses   += [float(v[0]) for v in learn.recorder.values]

    if gradual_unfreeze:
        # Phase 2: unfreeze all layers, use discriminative LRs
        # Early LSTM layers get lr_min; last LSTM layer + head get lr_max.
        tracker.epoch_train_losses = []
        learn.unfreeze()
        learn.fit_one_cycle(EPOCHS_UNFREEZE, LR_UNFREEZE, cbs=[tracker])
        train_losses += tracker.epoch_train_losses.copy()
        val_losses   += [float(v[0]) for v in learn.recorder.values]

    # ── Predictions on all three splits ───────────────────────────────────
    _, tr_prd_str = get_predictions(learn, train_data["text"].tolist())
    _, vl_prd_str = get_predictions(learn, val_df["text"].tolist())
    _, ts_prd_str = get_predictions(learn, test_df["text"].tolist())

    tr_lbl_str = train_data["label"].tolist()
    vl_lbl_str = val_df["label"].tolist()
    ts_lbl_str = test_df["label"].tolist()
    ts_ids     = test_df["id"].tolist()

    tr_met = compute_metrics(tr_lbl_str, tr_prd_str)
    vl_met = compute_metrics(vl_lbl_str, vl_prd_str)
    ts_met = compute_metrics(ts_lbl_str, ts_prd_str)

    print(f"\n  Train -- Acc {tr_met['accuracy']:.4f}  F1 {tr_met['f1_macro']:.4f}")
    print(f"  Val   -- Acc {vl_met['accuracy']:.4f}  F1 {vl_met['f1_macro']:.4f}")
    print(f"  Test  -- Acc {ts_met['accuracy']:.4f}  F1 {ts_met['f1_macro']:.4f}")

    # ── Save all outputs ───────────────────────────────────────────────────
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
    print(f"  All outputs saved to diagrams/{run_name}_*")

    return {
        "run":             run_name,
        "use_eda":         use_eda,
        "gradual_unfreeze":gradual_unfreeze,
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

    # ── Run 1: Frozen encoder, no EDA ────────────────────────────────────
    # Tests the pretrained AWD-LSTM representations as-is. Only the
    # classification head (concat-pool -> linear layers) is trained.
    r1 = run_experiment(
        run_name         = "run1_ulmfit_frozen",
        train_df         = train_df,
        val_df           = val_df,
        test_df          = test_df,
        use_eda          = False,
        gradual_unfreeze = False,
    )

    # ── Run 2: Frozen encoder + EDA ───────────────────────────────────────
    # Same as Run 1 but minority classes are oversampled via EDA before
    # training. Isolates the effect of augmentation from the effect of
    # encoder fine-tuning.
    r2 = run_experiment(
        run_name         = "run2_ulmfit_frozen_eda",
        train_df         = train_df,
        val_df           = val_df,
        test_df          = test_df,
        use_eda          = True,
        gradual_unfreeze = False,
    )

    # ── Run 3: Gradual unfreeze + EDA ─────────────────────────────────────
    # Full ULMFiT fine-tuning: frozen phase (same as Run 2) followed by
    # unfreezing all AWD-LSTM layers with discriminative learning rates
    # (early layers get LR_MIN, last layer + head get LR_MAX). Isolates
    # the added value of encoder fine-tuning over the frozen baseline.
    r3 = run_experiment(
        run_name         = "run3_ulmfit_gradual_unfreeze_eda",
        train_df         = train_df,
        val_df           = val_df,
        test_df          = test_df,
        use_eda          = True,
        gradual_unfreeze = True,
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
    print(f"{'Run':<42} | {'Acc':>6} | {'F1 Mac':>7} | {'F1 Wt':>7}")
    print(f"{'-'*42}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
    for r in all_results:
        t = r["test"]
        print(f"{r['run']:<42} | {t['accuracy']:>6.4f} | {t['f1_macro']:>7.4f} | {t['f1_weighted']:>7.4f}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Submit to CodaBench:  {zip_path}")
