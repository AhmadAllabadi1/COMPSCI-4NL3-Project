import os
import re
import time
import zipfile
import random
import numpy as np
import pandas as pd
import itertools

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("LightGBM available")
except ImportError:
    HAS_LGB = False
    print("LightGBM not found — install with: pip install lightgbm")
    print("Continuing without LightGBM branch.")

print("\n===== STARTING IMPROVED ENSEMBLE =====\n")

SEEDS        = [42, 77, 123]
DATA_DIR     = "../"
MODEL_NAME   = "intfloat/e5-large-v2"
USE_E5_PREFIX = True
REFIT_ON_FULL_DATA = True

TARGET_AUG_RATIO = 0.7
MIXUP_ALPHA      = 0.25

EMBED_CANDIDATE_C = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

WORD_MAX_FEATURES = 80000
WORD_NGRAM_RANGE  = (1, 3)
WORD_MIN_DF       = 1
WORD_MAX_DF       = 0.98

CHAR_MAX_FEATURES = 90000
CHAR_NGRAM_RANGE  = (3, 6)
CHAR_MIN_DF       = 1
CHAR_MAX_DF       = 0.99

SVC_CANDIDATE_C = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

LGB_PARAMS = {
    "objective":        "multiclass",
    "metric":           "multi_logloss",
    "n_estimators":     400,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "class_weight":     "balanced",
    "verbose":          -1,
    "n_jobs":           -1,
}

CLASS_BIAS = {
    "EMOTIONAL_SUPPORT": 0.04,
    "WARNING":           0.03,
}

print("SEEDS =", SEEDS)
print("MODEL =", MODEL_NAME)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro", zero_division=0),
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
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def apply_class_bias(probs, id2label, bias_map):
    probs = probs.copy()
    for class_id, class_name in id2label.items():
        if class_name in bias_map:
            probs[:, class_id] += bias_map[class_name]
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sums == 0, 1, row_sums)
    return probs

def augment_embeddings_with_mixup(X, y, id2label, alpha=0.25, target_ratio=0.7, seed=42):
    print("\n--- Running embedding mixup augmentation ---")
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y)

    class_counts = {cls: int((y == cls).sum()) for cls in np.unique(y)}
    max_count    = max(class_counts.values())
    target_count = int(max_count * target_ratio)

    X_aug, y_aug = [X], [y]

    print("Original class counts:")
    for cls in sorted(class_counts):
        print(f"  {id2label[cls]}: {class_counts[cls]}")
    print("Target count per class:", target_count)

    for cls in sorted(class_counts):
        idx   = np.where(y == cls)[0]
        count = len(idx)
        if count >= target_count:
            print(f"  Skipping {id2label[cls]} (already at/above target)")
            continue
        if count < 2:
            print(f"  Skipping {id2label[cls]} (not enough samples)")
            continue

        needed = target_count - count
        print(f"  Augmenting {id2label[cls]} with {needed} synthetic samples")
        synthetic = []
        for i in range(needed):
            i1, i2 = rng.choice(idx, size=2, replace=True)
            lam = rng.beta(alpha, alpha)
            v   = lam * X[i1] + (1.0 - lam) * X[i2]
            norm = np.linalg.norm(v)
            synthetic.append(v / norm if norm > 0 else v)
            if (i + 1) % 100 == 0 or i == needed - 1:
                print(f"    Generated {i+1}/{needed} for {id2label[cls]}")

        X_aug.append(np.asarray(synthetic))
        y_aug.append(np.full(needed, cls))

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)
    print("Augmented embedding shape:", X_out.shape)
    return X_out, y_out


def tune_thresholds(probs, y_true, n_classes):
    thresholds = np.ones(n_classes)
    for c in range(n_classes):
        best_t, best_f1 = 0.5, -1.0
        binary_true = (y_true == c).astype(int)
        for t in np.linspace(0.05, 0.90, 86):
            binary_pred = (probs[:, c] >= t).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds

def apply_thresholds(probs, thresholds):
    scaled = probs / (thresholds + 1e-9)
    return np.argmax(scaled, axis=1)


def encode_texts(encoder, texts, use_e5_prefix=False, batch_size=32):
    if use_e5_prefix:
        texts = ["query: " + t for t in texts]
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

print("\n===== LOADING DATA =====")

dataset = load_dataset(
    "csv",
    data_files={
        "train":      os.path.join(DATA_DIR, "train.csv"),
        "validation": os.path.join(DATA_DIR, "validation.csv"),
        "test":       os.path.join(DATA_DIR, "test.csv"),
    },
)

possible_text  = ["text", "comment", "comment_text", "body", "content"]
possible_label = ["label", "category", "category_level_1", "advice_type"]
possible_id    = ["id", "ID", "uid", "identifier"]

train_cols = dataset["train"].column_names
test_cols  = dataset["test"].column_names

TEXT_COL  = find_first_match(train_cols, possible_text)
LABEL_COL = find_first_match(train_cols, possible_label)
ID_COL    = find_first_match(test_cols,  possible_id)

if TEXT_COL is None:  raise ValueError(f"Text column not found. Columns: {train_cols}")
if LABEL_COL is None: raise ValueError(f"Label column not found. Columns: {train_cols}")

print("Text column:", TEXT_COL, "| Label column:", LABEL_COL)

train_df = pd.DataFrame(dataset["train"])
val_df   = pd.DataFrame(dataset["validation"])
test_df  = pd.DataFrame(dataset["test"])

for df in [train_df, val_df, test_df]:
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).apply(clean_text)

print("Shapes — train:", train_df.shape, "| val:", val_df.shape, "| test:", test_df.shape)

labels   = sorted(train_df[LABEL_COL].unique().tolist())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
n_classes = len(labels)

y_train = np.array([label2id[x] for x in train_df[LABEL_COL]])
y_val   = np.array([label2id[x] for x in val_df[LABEL_COL]])
y_test  = np.array([label2id[x] for x in test_df[LABEL_COL]])

train_texts = train_df[TEXT_COL].tolist()
val_texts   = val_df[TEXT_COL].tolist()
test_texts  = test_df[TEXT_COL].tolist()

print("Labels:", labels)

print("\n===== EXTRACTING SHARED FEATURES =====")

print(f"\nLoading encoder: {MODEL_NAME}")
encoder = SentenceTransformer(MODEL_NAME, device="cpu")

print("Encoding train...")
X_train_emb = encode_texts(encoder, train_texts, USE_E5_PREFIX)
print("Encoding val...")
X_val_emb   = encode_texts(encoder, val_texts,   USE_E5_PREFIX)
print("Encoding test...")
X_test_emb  = encode_texts(encoder, test_texts,  USE_E5_PREFIX)

print("Embedding shapes — train:", X_train_emb.shape, "| val:", X_val_emb.shape)

word_vec = TfidfVectorizer(
    lowercase=False, strip_accents="unicode",
    ngram_range=WORD_NGRAM_RANGE, min_df=WORD_MIN_DF,
    max_df=WORD_MAX_DF, max_features=WORD_MAX_FEATURES, sublinear_tf=True,
)
X_train_word = word_vec.fit_transform(train_texts)
X_val_word   = word_vec.transform(val_texts)
X_test_word  = word_vec.transform(test_texts)
print("Word TF-IDF train shape:", X_train_word.shape)

char_vec = TfidfVectorizer(
    lowercase=False, strip_accents="unicode", analyzer="char_wb",
    ngram_range=CHAR_NGRAM_RANGE, min_df=CHAR_MIN_DF,
    max_df=CHAR_MAX_DF, max_features=CHAR_MAX_FEATURES, sublinear_tf=True,
)
X_train_char = char_vec.fit_transform(train_texts)
X_val_char   = char_vec.transform(val_texts)
X_test_char  = char_vec.transform(test_texts)
print("Char TF-IDF train shape:", X_train_char.shape)

all_seed_val_probs  = []
all_seed_test_probs = []
seed_summaries      = []

for seed in SEEDS:
    print(f"\n\n{'='*16} SEED {seed} {'='*16}\n")
    random.seed(seed)
    np.random.seed(seed)

    X_train_emb_aug, y_train_emb_aug = augment_embeddings_with_mixup(
        X_train_emb, y_train, id2label=id2label,
        alpha=MIXUP_ALPHA, target_ratio=TARGET_AUG_RATIO, seed=seed,
    )

    print("\n--- Embedding LR branch ---")
    best_embed_f1, best_embed_c, best_embed_probs = -1, None, None

    for c in EMBED_CANDIDATE_C:
        model = LogisticRegression(
            C=c, max_iter=8000, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )
        model.fit(X_train_emb_aug, y_train_emb_aug)
        probs = apply_class_bias(model.predict_proba(X_val_emb), id2label, CLASS_BIAS)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        print(f"  C={c:5.2f}  val_f1={f1:.4f}")
        if f1 > best_embed_f1:
            best_embed_f1, best_embed_c, best_embed_model = f1, c, model
            best_embed_probs = probs

    print(f"  Best embed C={best_embed_c}  f1={best_embed_f1:.4f}")

    print("\n--- Word TF-IDF SVC branch ---")
    best_word_f1, best_word_c, best_word_probs = -1, None, None

    for c in SVC_CANDIDATE_C:
        base  = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=2000)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_word, y_train)
        probs = apply_class_bias(model.predict_proba(X_val_word), id2label, CLASS_BIAS)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        print(f"  C={c:5.2f}  val_f1={f1:.4f}")
        if f1 > best_word_f1:
            best_word_f1, best_word_c, best_word_model = f1, c, model
            best_word_probs = probs

    print(f"  Best word C={best_word_c}  f1={best_word_f1:.4f}")

    print("\n--- Char TF-IDF SVC branch ---")
    best_char_f1, best_char_c, best_char_probs = -1, None, None

    for c in SVC_CANDIDATE_C:
        base  = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=2000)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_char, y_train)
        probs = apply_class_bias(model.predict_proba(X_val_char), id2label, CLASS_BIAS)
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        print(f"  C={c:5.2f}  val_f1={f1:.4f}")
        if f1 > best_char_f1:
            best_char_f1, best_char_c, best_char_model = f1, c, model
            best_char_probs = probs

    print(f"  Best char C={best_char_c}  f1={best_char_f1:.4f}")

    lgb_val_probs = None
    if HAS_LGB:
        print("\n--- LightGBM branch ---")
        import lightgbm as lgb_mod
        lgb_model = lgb_mod.LGBMClassifier(**LGB_PARAMS, random_state=seed)
        lgb_model.fit(
            X_train_word, y_train,
            eval_set=[(X_val_word, y_val)],
            callbacks=[lgb_mod.early_stopping(30, verbose=False),
                       lgb_mod.log_evaluation(period=-1)],
        )
        lgb_val_probs = apply_class_bias(
            lgb_model.predict_proba(X_val_word), id2label, CLASS_BIAS
        )
        lgb_f1 = f1_score(y_val, np.argmax(lgb_val_probs, axis=1),
                           average="macro", zero_division=0)
        print(f"  LightGBM val_f1={lgb_f1:.4f}")

    print("\n--- Building stacking meta-learner (5-fold OOF) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_embed = np.zeros((len(y_train), n_classes))
    oof_word  = np.zeros((len(y_train), n_classes))
    oof_char  = np.zeros((len(y_train), n_classes))
    if HAS_LGB:
        oof_lgb = np.zeros((len(y_train), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_emb_aug[:len(y_train)], y_train)):
        m_emb = LogisticRegression(
            C=best_embed_c, max_iter=8000, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )
        m_emb.fit(X_train_emb[tr_idx], y_train[tr_idx])
        oof_embed[va_idx] = apply_class_bias(
            m_emb.predict_proba(X_train_emb[va_idx]), id2label, CLASS_BIAS
        )

        base = LinearSVC(C=best_word_c, class_weight="balanced",
                         random_state=seed, max_iter=2000)
        m_word = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        m_word.fit(X_train_word[tr_idx], y_train[tr_idx])
        oof_word[va_idx] = apply_class_bias(
            m_word.predict_proba(X_train_word[va_idx]), id2label, CLASS_BIAS
        )

        base = LinearSVC(C=best_char_c, class_weight="balanced",
                         random_state=seed, max_iter=2000)
        m_char = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        m_char.fit(X_train_char[tr_idx], y_train[tr_idx])
        oof_char[va_idx] = apply_class_bias(
            m_char.predict_proba(X_train_char[va_idx]), id2label, CLASS_BIAS
        )

        if HAS_LGB:
            import lightgbm as lgb_mod
            m_lgb = lgb_mod.LGBMClassifier(**LGB_PARAMS, random_state=seed)
            m_lgb.fit(
                X_train_word[tr_idx], y_train[tr_idx],
                callbacks=[lgb_mod.log_evaluation(period=-1)],
            )
            oof_lgb[va_idx] = apply_class_bias(
                m_lgb.predict_proba(X_train_word[va_idx]), id2label, CLASS_BIAS
            )

        print(f"  Fold {fold+1}/5 done")

    if HAS_LGB:
        oof_stack  = np.hstack([oof_embed, oof_word, oof_char, oof_lgb])
        val_stack  = np.hstack([best_embed_probs, best_word_probs,
                                best_char_probs,  lgb_val_probs])
    else:
        oof_stack = np.hstack([oof_embed, oof_word, oof_char])
        val_stack = np.hstack([best_embed_probs, best_word_probs, best_char_probs])

    meta_model = LogisticRegression(
        C=1.0, max_iter=4000, class_weight="balanced",
        solver="lbfgs", random_state=seed,
    )
    meta_model.fit(oof_stack, y_train)
    seed_val_probs = meta_model.predict_proba(val_stack)
    seed_val_probs = apply_class_bias(seed_val_probs, id2label, CLASS_BIAS)

    print("\n--- Tuning per-class thresholds on validation ---")
    thresholds = tune_thresholds(seed_val_probs, y_val, n_classes)
    print("  Optimal thresholds per class:", dict(zip(labels, thresholds.round(3))))

    seed_val_preds = apply_thresholds(seed_val_probs, thresholds)
    seed_val_metrics = print_metrics(f"STACKED ENSEMBLE — SEED {seed}", y_val, seed_val_preds, labels)

    if REFIT_ON_FULL_DATA:
        print(f"\n--- Refitting on full data (seed {seed}) ---")
        full_texts = train_texts + val_texts
        y_full     = np.concatenate([y_train, y_val])
        X_full_emb = encode_texts(encoder, full_texts, USE_E5_PREFIX)
        X_full_emb_aug, y_full_emb_aug = augment_embeddings_with_mixup(
            X_full_emb, y_full, id2label=id2label,
            alpha=MIXUP_ALPHA, target_ratio=TARGET_AUG_RATIO, seed=seed,
        )
        final_emb_model = LogisticRegression(
            C=best_embed_c, max_iter=8000, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )
        final_emb_model.fit(X_full_emb_aug, y_full_emb_aug)
        test_embed_probs = apply_class_bias(
            final_emb_model.predict_proba(X_test_emb), id2label, CLASS_BIAS
        )

        final_word_vec = TfidfVectorizer(
            lowercase=False, strip_accents="unicode",
            ngram_range=WORD_NGRAM_RANGE, min_df=WORD_MIN_DF,
            max_df=WORD_MAX_DF, max_features=WORD_MAX_FEATURES, sublinear_tf=True,
        )
        X_full_word      = final_word_vec.fit_transform(full_texts)
        X_test_word_full = final_word_vec.transform(test_texts)

        base = LinearSVC(C=best_word_c, class_weight="balanced",
                         random_state=seed, max_iter=2000)
        final_word_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        final_word_model.fit(X_full_word, y_full)
        test_word_probs = apply_class_bias(
            final_word_model.predict_proba(X_test_word_full), id2label, CLASS_BIAS
        )

        final_char_vec = TfidfVectorizer(
            lowercase=False, strip_accents="unicode", analyzer="char_wb",
            ngram_range=CHAR_NGRAM_RANGE, min_df=CHAR_MIN_DF,
            max_df=CHAR_MAX_DF, max_features=CHAR_MAX_FEATURES, sublinear_tf=True,
        )
        X_full_char      = final_char_vec.fit_transform(full_texts)
        X_test_char_full = final_char_vec.transform(test_texts)

        base = LinearSVC(C=best_char_c, class_weight="balanced",
                         random_state=seed, max_iter=2000)
        final_char_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        final_char_model.fit(X_full_char, y_full)
        test_char_probs = apply_class_bias(
            final_char_model.predict_proba(X_test_char_full), id2label, CLASS_BIAS
        )

        if HAS_LGB:
            import lightgbm as lgb_mod
            final_lgb = lgb_mod.LGBMClassifier(**LGB_PARAMS, random_state=seed)
            final_lgb.fit(X_full_word, y_full,
                          callbacks=[lgb_mod.log_evaluation(period=-1)])
            test_lgb_probs = apply_class_bias(
                final_lgb.predict_proba(X_test_word_full), id2label, CLASS_BIAS
            )
            test_stack = np.hstack([test_embed_probs, test_word_probs,
                                    test_char_probs,  test_lgb_probs])
        else:
            test_stack = np.hstack([test_embed_probs, test_word_probs, test_char_probs])

        if HAS_LGB:
            full_val_stack = np.hstack([best_embed_probs, best_word_probs,
                                        best_char_probs,  lgb_val_probs])
        else:
            full_val_stack = np.hstack([best_embed_probs, best_word_probs, best_char_probs])

        full_oof_stack = np.vstack([oof_stack, full_val_stack])
        y_meta_full    = np.concatenate([y_train, y_val])

        final_meta = LogisticRegression(
            C=1.0, max_iter=4000, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )
        final_meta.fit(full_oof_stack, y_meta_full)
        seed_test_probs = apply_class_bias(
            final_meta.predict_proba(test_stack), id2label, CLASS_BIAS
        )
    else:
        seed_test_probs = seed_val_probs

    all_seed_val_probs.append(seed_val_probs)
    all_seed_test_probs.append(seed_test_probs)

    seed_summaries.append({
        "seed":               seed,
        "embed_best_c":       best_embed_c,
        "word_best_c":        best_word_c,
        "char_best_c":        best_char_c,
        "val_f1_macro":       seed_val_metrics["f1_macro"],
        "val_accuracy":       seed_val_metrics["accuracy"],
        "thresholds":         str(dict(zip(labels, thresholds.round(3)))),
    })

print("\n\n===== FINAL AVERAGE ACROSS SEEDS =====")

avg_val_probs  = np.mean(np.stack(all_seed_val_probs,  axis=0), axis=0)
avg_test_probs = np.mean(np.stack(all_seed_test_probs, axis=0), axis=0)

print("\nTuning thresholds on averaged val probs...")
final_thresholds = tune_thresholds(avg_val_probs, y_val, n_classes)
print("Final thresholds:", dict(zip(labels, final_thresholds.round(3))))

val_preds   = apply_thresholds(avg_val_probs, final_thresholds)
val_metrics = print_metrics("FINAL AVERAGED ENSEMBLE ON VALIDATION", y_val, val_preds, labels)

test_preds       = apply_thresholds(avg_test_probs, final_thresholds)
test_pred_labels = [id2label[i] for i in test_preds]
test_metrics     = print_metrics("FINAL AVERAGED ENSEMBLE ON TEST", y_test, test_preds, labels)

print("\n===== SAVING FILES =====")

submission = pd.DataFrame({
    "id":    test_df[ID_COL].tolist() if ID_COL else list(range(len(test_df))),
    "label": test_pred_labels,
})
submission.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("predictions.csv")
print("Saved submission.zip")

val_out = pd.DataFrame({
    "id":         val_df[ID_COL].tolist() if ID_COL else list(range(len(val_df))),
    "true_label": [id2label[i] for i in y_val],
    "pred_label": [id2label[i] for i in val_preds],
})
val_out.to_csv("validation_predictions.csv", index=False)
print("Saved validation_predictions.csv")

pd.DataFrame(seed_summaries).to_csv("seed_summaries.csv", index=False)
print("Saved seed_summaries.csv")

summary = {
    "model":                MODEL_NAME,
    "seeds":                str(SEEDS),
    "validation_accuracy":  val_metrics["accuracy"],
    "validation_f1_macro":  val_metrics["f1_macro"],
    "test_accuracy":        test_metrics["accuracy"],
    "test_f1_macro":        test_metrics["f1_macro"],
    "final_thresholds":     str(dict(zip(labels, final_thresholds.round(3)))),
}
pd.DataFrame([summary]).to_csv("ensemble_summary.csv", index=False)
print("Saved ensemble_summary.csv")

print("\n===== DONE =====")
print("Files: predictions.csv | submission.zip | validation_predictions.csv")
print("       seed_summaries.csv | ensemble_summary.csv")