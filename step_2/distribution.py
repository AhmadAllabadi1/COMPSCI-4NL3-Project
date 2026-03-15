import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

VALID_LABELS = [
    "ADVICE",
    "WARNING",
    "EMOTIONAL_SUPPORT",
    "ANECDOTE",
    "APPRAISAL",
]


def main():
    input_file = "ground_truth.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    df = pd.read_csv(input_file)

    if "ground_truth" not in df.columns:
        print("Error: ground_truth column missing")
        sys.exit(1)

    counts = df["ground_truth"].value_counts()

    # Ensure all labels appear even if count = 0
    counts = counts.reindex(VALID_LABELS, fill_value=0)

    total = counts.sum()
    percentages = (counts / total * 100).round(2)

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print("\nLabel Distribution:")
    print(summary)

    print(f"\nTotal Samples: {total}")

    # Plot
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Ground Truth Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("label_distribution.png")

    print("\nSaved chart as label_distribution.png")


if __name__ == "__main__":
    main()