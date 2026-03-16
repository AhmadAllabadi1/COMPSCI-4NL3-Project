import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_label(value: str) -> str:
    return str(value).strip().upper()


def compute_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": report,
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2 baselines: random/majority baselines and TF-IDF + logistic regression."
    )
    parser.add_argument("--data", default=None, help="Path to labeled CSV file (will be split internally).")
    parser.add_argument("--train", default=None, help="Path to pre-split train CSV.")
    parser.add_argument("--val", default=None, help="Path to pre-split validation CSV (only used with --data).")
    parser.add_argument("--test", default=None, help="Path to pre-split test CSV.")
    parser.add_argument("--text-col", default="text", help="Name of text column.")
    parser.add_argument("--label-col", default="label1", help="Name of label column.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size (only used with --data).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="results/baselines",
        help="Directory where metrics, predictions, and summary are written.",
    )
    return parser.parse_args()


def load_split(path: str, text_col: str, label_col: str) -> tuple:
    df = pd.read_csv(path)
    missing = {text_col, label_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].map(normalize_label)
    return df[text_col], df[label_col]


def main() -> None:
    args = parse_args()

    if args.train and args.test:
        X_train, y_train = load_split(args.train, args.text_col, args.label_col)
        X_val, y_val = load_split(args.test, args.text_col, args.label_col)
        data_source = f"train={args.train}, test={args.test}"
    else:
        data_file = args.data or "merged_data.csv"
        df = pd.read_csv(data_file)
        missing = {args.text_col, args.label_col}.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        working = df[[args.text_col, args.label_col]].dropna(subset=[args.text_col, args.label_col]).copy()
        working[args.text_col] = working[args.text_col].astype(str)
        working[args.label_col] = working[args.label_col].map(normalize_label)
        X_train, X_val, y_train, y_val = train_test_split(
            working[args.text_col],
            working[args.label_col],
            test_size=args.val_size,
            random_state=args.random_state,
            stratify=working[args.label_col],
        )
        data_source = data_file

    class_counts = y_train.value_counts().sort_index()
    class_distribution = (class_counts / class_counts.sum()).to_dict()

    majority_label = y_train.value_counts().idxmax()
    majority_preds = np.full(shape=len(y_val), fill_value=majority_label)
    majority_metrics = compute_metrics(y_val, majority_preds)

    rng = np.random.default_rng(args.random_state)
    classes = y_train.value_counts().index.to_numpy()
    probs = (y_train.value_counts(normalize=True).loc[classes]).to_numpy()
    random_preds = rng.choice(classes, size=len(y_val), p=probs)
    random_metrics = compute_metrics(y_val, random_preds)

    simple_candidates = {
        "majority_baseline": majority_metrics,
        "random_baseline": random_metrics,
    }
    best_simple_name = max(simple_candidates, key=lambda name: simple_candidates[name]["accuracy"])

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)
    lr_preds = model.predict(X_val)
    lr_metrics = compute_metrics(y_val, lr_preds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "config": {
            "data": data_source,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "n_total": int(len(X_train)) + int(len(X_val)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        },
        "class_counts": class_counts.to_dict(),
        "class_distribution": class_distribution,
        "majority_baseline": majority_metrics,
        "random_baseline": random_metrics,
        "best_simple_baseline_by_accuracy": best_simple_name,
        "logreg_tfidf": lr_metrics,
    }
    save_json(out_dir / "validation_metrics.json", metrics_payload)

    preds_df = pd.DataFrame(
        {
            "text": X_val.values,
            "true_label": y_val.values,
            "pred_majority": majority_preds,
            "pred_random": random_preds,
            "pred_logreg_tfidf": lr_preds,
        }
    )
    preds_df.to_csv(out_dir / "validation_predictions.csv", index=False)

    summary_lines = [
        "Step 2 baseline summary",
        f"Data file: {data_source}",
        f"Text column: {args.text_col}",
        f"Label column: {args.label_col}",
        f"Total labeled rows used: {len(X_train) + len(X_val)}",
        f"Train/test split: {len(X_train)}/{len(X_val)}",
        "",
        f"Best simple baseline by validation accuracy: {best_simple_name}",
        f"- majority accuracy: {majority_metrics['accuracy']:.4f}, macro_f1: {majority_metrics['macro_f1']:.4f}",
        f"- random   accuracy: {random_metrics['accuracy']:.4f}, macro_f1: {random_metrics['macro_f1']:.4f}",
        "",
        "Trained baseline (TF-IDF + Logistic Regression):",
        f"- accuracy: {lr_metrics['accuracy']:.4f}",
        f"- macro_f1: {lr_metrics['macro_f1']:.4f}",
        f"- weighted_f1: {lr_metrics['weighted_f1']:.4f}",
        "",
        f"Outperforms best simple baseline (accuracy): {lr_metrics['accuracy'] > simple_candidates[best_simple_name]['accuracy']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote metrics to: {out_dir / 'validation_metrics.json'}")
    print(f"Wrote predictions to: {out_dir / 'validation_predictions.csv'}")
    print(f"Wrote summary to: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
