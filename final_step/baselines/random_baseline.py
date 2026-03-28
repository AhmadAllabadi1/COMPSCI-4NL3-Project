"""
Random Baseline — samples predictions from the training label distribution.

Usage:
    python random_baseline.py
"""

import numpy as np
from shared import load_data, evaluate_and_save, TRAIN_CSV, VAL_CSV, TEST_CSV, SEED


def main():
    print("=" * 60)
    print("  Random Baseline")
    print("=" * 60)

    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)
    test_df = load_data(TEST_CSV)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    classes = train_df["label"].value_counts().index.to_numpy()
    probs = train_df["label"].value_counts(normalize=True).loc[classes].to_numpy()
    print(f"  Class distribution: {dict(zip(classes, probs.round(3)))}")

    rng = np.random.default_rng(SEED)

    for split_name, df in [("val", val_df), ("test", test_df)]:
        preds = rng.choice(classes, size=len(df), p=probs)
        m = evaluate_and_save("random", df["label"], preds, df["text"].tolist(), split_name)
        print(f"\n  {split_name.upper()}: accuracy={m['accuracy']:.4f}, macro_f1={m['f1_macro']:.4f}, "
              f"weighted_f1={m['f1_weighted']:.4f}")

    print("\n" + "=" * 60)
    print("  Done! Check baselines/results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
