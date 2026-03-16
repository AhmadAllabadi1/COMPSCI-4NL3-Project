import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def normalize_label(value: str) -> str:
    return str(value).strip().upper()


def compute_metrics(y_true, y_pred) -> None:
    print(f"  accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"  balanced_accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  macro_f1:          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted_f1:       {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--val", default="validation.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output", default="submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train).dropna(subset=[args.text_col, args.label_col])
    val_df = pd.read_csv(args.val).dropna(subset=[args.text_col, args.label_col])
    test_df = pd.read_csv(args.test)

    train_df[args.label_col] = train_df[args.label_col].map(normalize_label)
    val_df[args.label_col] = val_df[args.label_col].map(normalize_label)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(train_df[args.text_col], train_df[args.label_col])

    val_preds = model.predict(val_df[args.text_col])
    print("\nValidation set metrics:")
    compute_metrics(val_df[args.label_col].values, val_preds)

    test_preds = model.predict(test_df[args.text_col])
    submission = pd.DataFrame({"id": test_df["id"].values, "label": test_preds})
    submission.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
