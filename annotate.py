import json
import csv
import os

DATA_FILE = "structured_data.json"
OUTPUT_FILE = "annotations.csv"

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
