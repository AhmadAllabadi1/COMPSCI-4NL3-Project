import argparse
import os
import sys
import pandas as pd
from pandas.errors import EmptyDataError

VALID_LABELS = [
    "ADVICE",
    "WARNING",
    "EMOTIONAL_SUPPORT",
    "ANECDOTE",
    "APPRAISAL",
]

REQUIRED_COLUMNS = ["id", "text", "label1", "label2", "annotator1", "annotator2"]


def normalize_label(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def check_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print("Error: Missing required columns:", ", ".join(missing))
        sys.exit(1)


def build_option_map(voted_labels):
    ordered_labels = []

    for label in voted_labels:
        if label and label in VALID_LABELS and label not in ordered_labels:
            ordered_labels.append(label)

    for label in VALID_LABELS:
        if label not in ordered_labels:
            ordered_labels.append(label)

    option_map = {}
    for i, label in enumerate(ordered_labels, start=1):
        option_map[str(i)] = label

    return option_map


def prompt_for_label(row):
    label1 = normalize_label(row["label1"])
    label2 = normalize_label(row["label2"])

    print("\n" + "=" * 80)
    print(f"ID: {row['id']}")
    print("-" * 80)
    print("TEXT:")
    print(row["text"])
    print("-" * 80)
    print(f"label1 ({row['annotator1']}): {label1 if label1 else '[MISSING]'}")
    print(f"label2 ({row['annotator2']}): {label2 if label2 else '[MISSING]'}")
    print("-" * 80)
    print("Choose the final ground-truth label:")

    voted_labels = []
    if label1 in VALID_LABELS:
        voted_labels.append(label1)
    if label2 in VALID_LABELS:
        voted_labels.append(label2)

    option_map = build_option_map(voted_labels)

    for num, label in option_map.items():
        print(f"{num}. {label}")

    print("Type the number or the label name directly.")

    while True:
        choice = input("Your choice: ").strip().upper()

        if choice in option_map:
            return option_map[choice]

        if choice in VALID_LABELS:
            return choice

        print("Invalid choice. Please enter a valid number or label name.")


def save_progress(df, output_path):
    df.to_csv(output_path, index=False)


def load_existing_output(output_path, input_df):
    if not os.path.exists(output_path):
        return None

    if os.path.getsize(output_path) == 0:
        print(f"Warning: {output_path} is empty. Starting a fresh run.")
        return None

    try:
        existing_df = pd.read_csv(output_path)
    except EmptyDataError:
        print(f"Warning: {output_path} is unreadable/empty. Starting a fresh run.")
        return None
    except Exception as e:
        print(f"Warning: Could not read {output_path} ({e}). Starting a fresh run.")
        return None

    if "ground_truth" not in existing_df.columns:
        print(f"Warning: {output_path} has no ground_truth column. Starting a fresh run.")
        return None

    if len(existing_df) != len(input_df):
        print(f"Warning: {output_path} row count does not match input. Starting a fresh run.")
        return None

    return existing_df


def main():
    parser = argparse.ArgumentParser(description="Create ground-truth labels from 2 annotators.")
    parser.add_argument("input_csv", help="Path to input annotations CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="ground_truth.csv",
        help="Path to output CSV (default: ground_truth.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: File not found: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    check_columns(df)

    df["label1"] = df["label1"].apply(normalize_label)
    df["label2"] = df["label2"].apply(normalize_label)

    existing_df = load_existing_output(args.output, df)

    if existing_df is not None:
        df = existing_df
        df["label1"] = df["label1"].apply(normalize_label)
        df["label2"] = df["label2"].apply(normalize_label)
        df["ground_truth"] = df["ground_truth"].apply(normalize_label)
        print(f"Resuming from existing file: {args.output}")
    else:
        df["ground_truth"] = ""
        save_progress(df, args.output)

    agreement_count = 0
    disagreement_count = 0
    single_vote_count = 0
    manual_other_count = 0

    for idx, row in df.iterrows():
        existing_ground = normalize_label(row.get("ground_truth", ""))

        if existing_ground in VALID_LABELS:
            continue

        label1 = normalize_label(row["label1"])
        label2 = normalize_label(row["label2"])

        label1_valid = label1 in VALID_LABELS
        label2_valid = label2 in VALID_LABELS

        # both valid and same
        if label1_valid and label2_valid and label1 == label2:
            df.at[idx, "ground_truth"] = label1
            agreement_count += 1
            save_progress(df, args.output)
            continue

        # only one valid vote exists -> automatically use it
        if label1_valid and not label2_valid:
            df.at[idx, "ground_truth"] = label1
            single_vote_count += 1
            save_progress(df, args.output)
            continue

        if label2_valid and not label1_valid:
            df.at[idx, "ground_truth"] = label2
            single_vote_count += 1
            save_progress(df, args.output)
            continue

        # both valid but disagree
        if label1_valid and label2_valid and label1 != label2:
            final_label = prompt_for_label(row)
            df.at[idx, "ground_truth"] = final_label
            disagreement_count += 1
            save_progress(df, args.output)
            continue

        # both missing or invalid
        print("\n" + "=" * 80)
        print(f"ID: {row['id']}")
        print("No valid annotator vote found for this row.")
        print(f"label1: '{label1}'")
        print(f"label2: '{label2}'")

        final_label = prompt_for_label(row)
        df.at[idx, "ground_truth"] = final_label
        manual_other_count += 1
        save_progress(df, args.output)

    completed_ground_truth = df["ground_truth"].apply(normalize_label)
    completed_count = completed_ground_truth.isin(VALID_LABELS).sum()

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Total rows: {len(df)}")
    print(f"Completed rows: {completed_count}")
    print(f"Automatic agreements: {agreement_count}")
    print(f"Automatic single-vote labels: {single_vote_count}")
    print(f"Manual disagreements: {disagreement_count}")
    print(f"Manual rows with no valid votes: {manual_other_count}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()