import os, re, zipfile, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true,y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true,y_pred,average="macro",zero_division=0),
        "recall_macro": recall_score(y_true, y_pred , average="macro", zero_division=0),
    }

def print_metrics(title, y_true, y_pred, labels):
    print(f"\n===== {title} =====")
    m = compute_metrics(y_true, y_pred)
    print("Metrics:", m)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_true, y_pred))
    return m

def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true,y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        title=title, ylabel="True label", xlabel="Predicted label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

def save_per_class_f1_chart(y_true, y_pred, labels, title, filename):
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    vals = [report[label]["f1-score"] for label in labels]

    plt.figure(figsize=(8,5))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

def find_first_match(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def clean_text(text):
    text = str(text)
    text = re.sub(r"(?m)^>.*$", " REDDITQUOTE ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"u/\w+", " REDDITOR ", text)
    text = re.sub(r"r/\w+", " SUBREDDIT ", text)
    text = re.sub(r"\s+"," ", text).strip()
    return text.lower()


def apply_class_bias(probs, id2label):
    probs = probs.copy()
    bias_map = {"EMOTIONAL_SUPPORT": 0.04, "WARNING":0.03}
    for class_id, class_name in id2label.items():
        if class_name in bias_map:
            probs[:, class_id] += bias_map[class_name]
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row_sums == 0, 1, row_sums)


def augment_embeddings_with_mixup(X, y, id2label, seed):
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y)

    counts = {cls: int((y == cls).sum()) for cls in np.unique(y)}
    target_count = int(max(counts.values()) * 0.7)

    X_aug, y_aug = [X], [y]

    for cls in sorted(counts):
        idx = np.where(y == cls)[0]
        count = len(idx)
        if count >= target_count or count < 2:
            continue

        needed = target_count - count
        synthetic = []
        for _ in range(needed):
            i1, i2 = rng.choice(idx, size=2, replace=True)
            lam = rng.beta(0.25, 0.25)
            v = lam * X[i1] + (1.0 - lam) * X[i2]
            norm = np.linalg.norm(v)
            synthetic.append(v / norm if norm > 0 else v)

        X_aug.append(np.asarray(synthetic))
        y_aug.append(np.full(needed, cls))

    return np.vstack(X_aug), np.concatenate(y_aug)


def tune_thresholds(probs, y_true, n_classes):
    thresholds = np.ones(n_classes)
    for c in range(n_classes):
        best_t, best_f1 = 0.5, -1.0
        binary_true = (y_true == c).astype(int)
        for t in np.linspace(0.05, 0.90, 86):
            binary_pred = (probs[:,c] >= t).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
            if f1 > best_f1:
                best_f1 , best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def apply_thresholds(probs, thresholds):
    return np.argmax(probs / (thresholds + 1e-9), axis=1)


def encode_texts(encoder, texts, use_prefix=True, batch_size=32):
    if use_prefix:
        texts = ["query: " + t for t in texts]
    return encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join("..", "train.csv"),
        "validation": os.path.join("..", "validation.csv"),
        "test": os.path.join("..", "test.csv"),
    },
)

text_col  = find_first_match(dataset["train"].column_names, ["text", "comment", "comment_text", "body", "content"])
label_col = find_first_match(dataset["train"].column_names, ["label", "category", "category_level_1", "advice_type"])
id_col = find_first_match(dataset["test"].column_names, ["id", "ID", "uid", "identifier"])

if text_col is None:
    raise ValueError(f"Text column not found. Columns: {dataset['train'].column_names}")
if label_col is None:
    raise ValueError(f"Label column not found. Columns: {dataset['train'].column_names}")

train_df = pd.DataFrame(dataset["train"])
val_df   = pd.DataFrame(dataset["validation"])
test_df  = pd.DataFrame(dataset["test"])

for df in [train_df, val_df, test_df]:
    df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)

labels = sorted(train_df[label_col].unique().tolist())
label2id = {label:i for i, label in enumerate(labels)}
id2label = {i:label for label, i in label2id.items()}
n_classes = len(labels)

y_train = np.array([label2id[x] for x in train_df[label_col]])
y_val   = np.array([label2id[x] for x in val_df[label_col]])
y_test  = np.array([label2id[x] for x in test_df[label_col]])

train_texts = train_df[text_col].tolist()
val_texts   = val_df[text_col].tolist()
test_texts  = test_df[text_col].tolist()

encoder = SentenceTransformer("intfloat/e5-large-v2", device="cpu")

X_train_emb = encode_texts(encoder, train_texts)
X_val_emb   = encode_texts(encoder, val_texts)
X_test_emb  = encode_texts(encoder, test_texts)

word_vec = TfidfVectorizer(
    lowercase=False, strip_accents="unicode", ngram_range=(1,3),
    min_df=1, max_df=0.98, max_features=80000, sublinear_tf=True
)
X_train_word = word_vec.fit_transform(train_texts)
X_val_word   = word_vec.transform(val_texts)

char_vec = TfidfVectorizer(
    lowercase=False, strip_accents="unicode", analyzer="char_wb",
    ngram_range=(3,6), min_df=1, max_df=0.99, max_features=90000, sublinear_tf=True
)
X_train_char = char_vec.fit_transform(train_texts)
X_val_char   = char_vec.transform(val_texts)

all_seed_val_probs, all_seed_test_probs, seed_summaries = [], [], []

for seed in [42, 77, 123]:
    random.seed(seed)
    np.random.seed(seed)

    X_train_emb_aug, y_train_emb_aug = augment_embeddings_with_mixup(X_train_emb, y_train, id2label, seed)

    best_embed_f1, best_embed_c, best_embed_probs = -1, None, None
    for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        model = LogisticRegression(C=c, max_iter=8000, class_weight="balanced", solver="lbfgs", random_state=seed)
        model.fit(X_train_emb_aug, y_train_emb_aug)
        probs = apply_class_bias(model.predict_proba(X_val_emb), id2label)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        if f1 > best_embed_f1:
            best_embed_f1, best_embed_c, best_embed_probs = f1, c, probs

    best_word_f1, best_word_c, best_word_probs = -1, None, None
    for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        base = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=2000)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_word, y_train)
        probs = apply_class_bias(model.predict_proba(X_val_word), id2label)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        if f1 > best_word_f1:
            best_word_f1 , best_word_c, best_word_probs = f1, c, probs

    best_char_f1, best_char_c, best_char_probs = -1, None, None
    for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        base = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=2000)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_char, y_train)
        probs = apply_class_bias(model.predict_proba(X_val_char), id2label)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        if f1 > best_char_f1:
            best_char_f1, best_char_c, best_char_probs = f1, c, probs

    oof_embed = np.zeros((len(y_train), n_classes))
    oof_word  = np.zeros((len(y_train), n_classes))
    oof_char  = np.zeros((len(y_train), n_classes))

    for tr_idx, va_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train_emb, y_train):
        m_emb = LogisticRegression(C=best_embed_c, max_iter=8000, class_weight="balanced", solver="lbfgs", random_state=seed)
        m_emb.fit(X_train_emb[tr_idx], y_train[tr_idx])
        oof_embed[va_idx] = apply_class_bias(m_emb.predict_proba(X_train_emb[va_idx]), id2label)

        base = LinearSVC(C=best_word_c, class_weight="balanced", random_state=seed, max_iter=2000)
        m_word = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        m_word.fit(X_train_word[tr_idx], y_train[tr_idx])
        oof_word[va_idx] = apply_class_bias(m_word.predict_proba(X_train_word[va_idx]), id2label)

        base = LinearSVC(C=best_char_c, class_weight="balanced",random_state=seed,max_iter=2000)
        m_char = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        m_char.fit(X_train_char[tr_idx], y_train[tr_idx])
        oof_char[va_idx] = apply_class_bias(m_char.predict_proba(X_train_char[va_idx]), id2label)

    oof_stack = np.hstack([oof_embed, oof_word, oof_char])
    val_stack = np.hstack([best_embed_probs, best_word_probs, best_char_probs])

    meta_model = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    meta_model.fit(oof_stack, y_train)
    seed_val_probs = apply_class_bias(meta_model.predict_proba(val_stack), id2label)

    thresholds = tune_thresholds(seed_val_probs, y_val, n_classes)
    seed_val_preds = apply_thresholds(seed_val_probs, thresholds)
    seed_val_metrics = print_metrics(f"STACKED ENSEMBLE — SEED {seed}", y_val, seed_val_preds, labels)

    full_texts = train_texts + val_texts
    y_full = np.concatenate([y_train, y_val])

    X_full_emb = encode_texts(encoder, full_texts)
    X_full_emb_aug, y_full_emb_aug = augment_embeddings_with_mixup(X_full_emb, y_full, id2label, seed)

    final_emb_model = LogisticRegression(C=best_embed_c, max_iter=8000, class_weight="balanced", solver="lbfgs", random_state=seed)
    final_emb_model.fit(X_full_emb_aug, y_full_emb_aug)
    test_embed_probs = apply_class_bias(final_emb_model.predict_proba(X_test_emb), id2label)

    final_word_vec = TfidfVectorizer(
            lowercase=False, strip_accents="unicode", ngram_range=(1,3),
            min_df=1, max_df=0.98, max_features=80000, sublinear_tf=True
    )
    X_full_word = final_word_vec.fit_transform(full_texts)
    X_test_word_full = final_word_vec.transform(test_texts)

    base = LinearSVC(C=best_word_c, class_weight="balanced", random_state=seed, max_iter=2000)
    final_word_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    final_word_model.fit(X_full_word, y_full)
    test_word_probs = apply_class_bias(final_word_model.predict_proba(X_test_word_full), id2label)

    final_char_vec = TfidfVectorizer(
            lowercase=False, strip_accents="unicode", analyzer="char_wb",
            ngram_range=(3,6), min_df=1, max_df=0.99, max_features=90000, sublinear_tf=True
    )
    X_full_char = final_char_vec.fit_transform(full_texts)
    X_test_char_full = final_char_vec.transform(test_texts)

    base = LinearSVC(C=best_char_c, class_weight="balanced", random_state=seed, max_iter=2000)
    final_char_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    final_char_model.fit(X_full_char, y_full)
    test_char_probs = apply_class_bias(final_char_model.predict_proba(X_test_char_full), id2label)

    test_stack = np.hstack([test_embed_probs, test_word_probs, test_char_probs])
    full_val_stack = np.hstack([best_embed_probs, best_word_probs, best_char_probs])
    full_oof_stack = np.vstack([oof_stack, full_val_stack])
    y_meta_full = np.concatenate([y_train, y_val])

    final_meta = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    final_meta.fit(full_oof_stack, y_meta_full)
    seed_test_probs = apply_class_bias(final_meta.predict_proba(test_stack), id2label)
    

    all_seed_val_probs.append(seed_val_probs)
    all_seed_test_probs.append(seed_test_probs)

    seed_summaries.append({
        "seed": seed,
        "embed_best_c": best_embed_c,
        "word_best_c": best_word_c,
        "char_best_c": best_char_c,
        "val_f1_macro": seed_val_metrics["f1_macro"],
        "val_accuracy": seed_val_metrics["accuracy"],
        "thresholds": str(dict(zip(labels, thresholds.round(3))))
    })

avg_val_probs = np.mean(np.stack(all_seed_val_probs, axis=0), axis=0)
avg_test_probs = np.mean(np.stack(all_seed_test_probs, axis=0), axis=0)

final_thresholds = tune_thresholds(avg_val_probs, y_val, n_classes)

val_preds = apply_thresholds(avg_val_probs, final_thresholds)
val_metrics = print_metrics("FINAL AVERAGED ENSEMBLE ON VALIDATION", y_val, val_preds, labels)

test_preds = apply_thresholds(avg_test_probs, final_thresholds)
test_pred_labels = [id2label[i] for i in test_preds]
test_metrics = print_metrics("FINAL AVERAGED ENSEMBLE ON TEST", y_test, test_preds, labels)

save_confusion_matrix(y_test, test_preds, labels, "Final Averaged Ensemble Confusion Matrix (Test)", "final_test_confusion_matrix.png")
save_per_class_f1_chart(y_test, test_preds, labels, "Final Averaged Ensemble Per-Class F1 (Test)", "final_test_per_class_f1.png")

submission = pd.DataFrame({
    "id": test_df[id_col].tolist() if id_col else list(range(len(test_df))),
    "label": test_pred_labels,
})
submission.to_csv("predictions.csv", index=False)

with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("predictions.csv")

pd.DataFrame(seed_summaries).to_csv("seed_summaries.csv", index=False)

pd.DataFrame([{
    "model": "intfloat/e5-large-v2",
    "seeds": str([42, 77, 123]),
    "validation_accuracy": val_metrics["accuracy"],
    "validation_f1_macro": val_metrics["f1_macro"],
    "validation_precision_macro": val_metrics["precision_macro"],
    "validation_recall_macro": val_metrics["recall_macro"],
    "test_accuracy": test_metrics["accuracy"],
    "test_f1_macro": test_metrics["f1_macro"],
    "test_precision_macro": test_metrics["precision_macro"],
    "test_recall_macro": test_metrics["recall_macro"],
    "final_thresholds": str(dict(zip(labels, final_thresholds.round(3)))),
}]).to_csv("ensemble_summary.csv", index=False)
