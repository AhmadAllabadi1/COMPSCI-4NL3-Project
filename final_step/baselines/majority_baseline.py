"""
Majority Baseline — always predicts the most frequent training label.

Usage:
    python majority_baseline.py
"""

import numpy as np
from shared import load_data, evaluate_and_save, TRAIN_CSV, VAL_CSV, TEST_CSV


def main():
    print("=" * 60)
    print("  Majority Baseline")
    print("=" * 60)

    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)
    test_df = load_data(TEST_CSV)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    majority_label = train_df["label"].value_counts().idxmax()
    print(f"  Majority label: {majority_label}")

    for split_name, df in [("val", val_df), ("test", test_df)]:
        preds = np.full(len(df), majority_label)
        m = evaluate_and_save("majority", df["label"], preds, df["text"].tolist(), split_name)
        print(f"\n  {split_name.upper()}: accuracy={m['accuracy']:.4f}, macro_f1={m['f1_macro']:.4f}, "
              f"weighted_f1={m['f1_weighted']:.4f}")

    print("\n" + "=" * 60)
    print("  Done! Check baselines/results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
