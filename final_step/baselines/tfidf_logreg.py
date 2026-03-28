"""
TF-IDF + Logistic Regression Baseline

Vectorizes text with TF-IDF (unigrams + bigrams), classifies with LogReg.

Usage:
    python tfidf_logreg.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from shared import load_data, evaluate_and_save, TRAIN_CSV, VAL_CSV, TEST_CSV, SEED


def main():
    print("=" * 60)
    print("  TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)
    test_df = load_data(TEST_CSV)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train
    print("\n  Training TF-IDF + LogReg...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    model.fit(train_df["text"], train_df["label"])

    # Get vocab size
    vocab_size = len(model.named_steps["tfidf"].vocabulary_)
    print(f"  Vocabulary size: {vocab_size} features")

    # Evaluate on both splits
    for split_name, df in [("val", val_df), ("test", test_df)]:
        preds = model.predict(df["text"])
        m = evaluate_and_save("tfidf_logreg", df["label"], preds, df["text"].tolist(), split_name)
        print(f"\n  {split_name.upper()}: accuracy={m['accuracy']:.4f}, macro_f1={m['f1_macro']:.4f}, "
              f"weighted_f1={m['f1_weighted']:.4f}")

    print("\n" + "=" * 60)
    print("  Done! Check baselines/results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
