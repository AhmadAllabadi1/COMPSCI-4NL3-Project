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

# patterns that indicate redirect/meta comments (excluded from dataset)
REDIRECT_PATTERNS = [
    "i am a bot",
    "performed automatically",
    "your comment has been removed",
    "your post has been removed",
    "your submission has been removed",
    "what's your question",
    "read the sidebar",
    "r/legaladvice",
    "/r/legaladvice",
    "r/personalfinance",
    "/r/personalfinance",
]


def is_redirect_or_meta(text: str) -> bool:
    """Return True if the comment is a redirect, bot message, or meta-question."""
    if not text or not isinstance(text, str):
        return False
    lower = text.lower()
    return any(p in lower for p in REDIRECT_PATTERNS)


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
            if is_redirect_or_meta(body):
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

# split into 4 files of 500 each
OUTPUT_FILES = [
    "ahmad_data.json",
    "karam_data.json",
    "rayan_data.json",
    "aizaz_data.json",
]
CHUNK_SIZE = 500

for i, filename in enumerate(OUTPUT_FILES):
    start = i * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk = all_data[start:end]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
    print(f"Created {filename} with {len(chunk)} instances.")

print("DONE. Created 4 files with 500 instances each.")
