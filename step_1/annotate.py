"""
Annotate Reddit comments with support/response labels.

Usage:
  python annotate.py <data_file>
  python annotate.py ahmad_data.json

Input:  JSON file with [{"id": N, "text": "..."}, ...]
Output: CSV file (e.g. ahmad_data.json -> ahmad_annotations.csv)
        Columns: id, text, label (ADVICE | WARNING | EMOTIONAL_SUPPORT | ANECDOTE | APPRAISAL)
"""
import json
import csv
import os
import sys

# accept filename as command-line argument or prompt for it
if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]
else:
    DATA_FILE = input("Enter data file to annotate (e.g. ahmad_data.json): ").strip()

if not DATA_FILE:
    print("No file specified. Exiting.")
    sys.exit(1)

# derive output file from input
if DATA_FILE.endswith("_data.json"):
    OUTPUT_FILE = DATA_FILE.replace("_data.json", "_annotations.csv")
else:
    base = os.path.splitext(DATA_FILE)[0]
    OUTPUT_FILE = base + "_annotations.csv"

if not os.path.exists(DATA_FILE):
    print(f"File not found: {DATA_FILE}")
    sys.exit(1)

LABELS = {
    "1": "ADVICE",
    "2": "WARNING",
    "3": "EMOTIONAL_SUPPORT",
    "4": "ANECDOTE",
    "5": "APPRAISAL"
}

# load data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# load previous annotations if exist
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(int(row["id"]))

# open file for appending
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["id", "text", "label"])

    for item in data:
        if item["id"] in completed:
            continue

        print("\nComment", item["id"], "/", len(data))
        print("-" * 40)
        print(item["text"])
        print("-" * 40)

        print("1 = ADVICE")
        print("2 = WARNING")
        print("3 = EMOTIONAL_SUPPORT")
        print("4 = ANECDOTE")
        print("5 = APPRAISAL")

        label = None
        while label not in LABELS:
            label = input("Label: ").strip()

        writer.writerow([item["id"], item["text"], LABELS[label]])
        f.flush()

print("All done!")
