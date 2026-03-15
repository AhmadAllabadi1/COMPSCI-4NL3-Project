import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

VALID_LABELS = [
    "ADVICE",
    "WARNING",
    "EMOTIONAL_SUPPORT",
    "ANECDOTE",
    "APPRAISAL",
]


def print_split_distribution(name, df):
    counts = df["ground_truth"].value_counts().reindex(VALID_LABELS, fill_value=0)
    total = len(df)
    percentages = (counts / total * 100).round(2)

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print("\n" + "=" * 60)
    print(f"{name} Distribution (Total = {total})")
    print(summary)


def main():
    input_file = "ground_truth.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    df = pd.read_csv(input_file)

    if "ground_truth" not in df.columns:
        print("Error: ground_truth column missing")
        sys.exit(1)

    # 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["ground_truth"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["ground_truth"],
        random_state=42
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("Saved:")
    print(f"train.csv: {len(train_df)} rows")
    print(f"validation.csv: {len(val_df)} rows")
    print(f"test.csv: {len(test_df)} rows")

    print_split_distribution("Train", train_df)
    print_split_distribution("Validation", val_df)
    print_split_distribution("Test", test_df)


if __name__ == "__main__":
    main()