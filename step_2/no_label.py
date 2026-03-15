import pandas as pd
import os
import sys


def main():
    input_file = "test.csv"
    output_file = "test_unlabeled.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    df = pd.read_csv(input_file)

    required_columns = ["id", "text"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print("Error: Missing required columns:", ", ".join(missing))
        sys.exit(1)

    test_unlabeled = df[["id", "text"]]
    test_unlabeled.to_csv(output_file, index=False)

    print(f"Created {output_file} with {len(test_unlabeled)} rows.")
    print("Columns kept: id, text")


if __name__ == "__main__":
    main()