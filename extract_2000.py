"""
This script is used to extract 2000 comments from the jsonl files.
"""
import json
import random
import glob
import os

# set seed for reproducibility
random.seed(42)

# total comments to extract
TOTAL_NEEDED = 2000

# find all jsonl files in folder
files = glob.glob(os.path.join("raw_data", "*.jsonl"))

# first pass: collect all valid lines (exclude [removed] and [deleted])
all_valid_lines = []
for file in files:
    print("Reading:", file)
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        try:
            obj = json.loads(line)
            body = obj.get("body", obj.get("text", ""))
            if body in ("[removed]", "[deleted]"):
                continue
            all_valid_lines.append(line)
        except Exception:
            continue

# sample 2000 from the combined pool
n = min(TOTAL_NEEDED, len(all_valid_lines))
sampled_lines = random.sample(all_valid_lines, n)

all_data = []
current_id = 1
for line in sampled_lines:
    try:
        obj = json.loads(line)
        text = obj["body"]

        entry = {
            "id": current_id,
            "text": text
        }

        all_data.append(entry)
        current_id += 1

    except Exception:
        continue

# save final dataset
with open("structured_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print("DONE. Created structured_data.json with", len(all_data), "instances.")
