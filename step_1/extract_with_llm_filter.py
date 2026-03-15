"""
Extract 2000 comments from raw jsonl files using LLM pre-filter.
Comments are classified as KEEP, DROP, or UNCLEAR.
KEEP and UNCLEAR are added to the dataset; DROP is excluded.

Usage: python extract_with_llm_filter.py
"""
import json
import random
import glob
import os
import time
from openai import OpenAI

# set seed for reproducibility
random.seed(42)

TOTAL_NEEDED = 2000
MODEL = "gpt-4o-mini"
DELAY_BETWEEN_CALLS = 0.2  # seconds, to avoid rate limits

# put your API key here
OPENAI_API_KEY = "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"

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

CLASSIFY_PROMPT = """You are classifying Reddit comments for an advice/support dataset. Each comment will be labeled for annotation into one of: Advice, Warning, Emotional Support, Anecdote, or Appraisal.

Respond with exactly one word:
- KEEP: The comment is useful advice/support content. It fits one of the 5 categories (advice, warning, emotional support, anecdote, appraisal) or is clearly helpful.
- DROP: Garbage. Off-topic, spam, bot message, mod removal, low-effort (e.g. "lol", "this"), incomprehensible, or clearly not advice/support.
- UNCLEAR: Borderline. Could go either way; include for human judgment.

Respond with ONLY: KEEP, DROP, or UNCLEAR"""


def is_redirect_or_meta(text: str) -> bool:
    """Return True if the comment is a redirect, bot message, or meta-question."""
    if not text or not isinstance(text, str):
        return False
    lower = text.lower()
    return any(p in lower for p in REDIRECT_PATTERNS)


def classify_comment(client: OpenAI, text: str, max_retries: int = 3) -> str:
    """Call LLM to classify comment. Returns KEEP, DROP, or UNCLEAR."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": CLASSIFY_PROMPT},
                    {"role": "user", "content": f"Comment to classify:\n\n{text[:2000]}"},
                ],
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().upper()
            # take first word in case model adds explanation
            word = result.split()[0] if result else ""
            if word in ("KEEP", "DROP", "UNCLEAR"):
                return word
            # fallback: if model said something like "KEEP - this is advice"
            for label in ("KEEP", "DROP", "UNCLEAR"):
                if label in result:
                    return label
            return "UNCLEAR"  # default when parse fails
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise e
    return "UNCLEAR"


# load raw jsonl files
files = glob.glob(os.path.join("raw_data", "*.jsonl"))
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

# shuffle for reproducibility
random.shuffle(all_valid_lines)

# LLM filter: iterate until we have 2000
client = OpenAI(api_key=OPENAI_API_KEY)
all_data = []
current_id = 1
dropped = 0
kept = 0
unclear = 0

print(f"Filtering with LLM (model={MODEL}). Need {TOTAL_NEEDED} comments...")
for i, line in enumerate(all_valid_lines):
    if len(all_data) >= TOTAL_NEEDED:
        break

    try:
        obj = json.loads(line)
        text = obj.get("body", obj.get("text", ""))
    except Exception:
        continue

    label = classify_comment(client, text)
    time.sleep(DELAY_BETWEEN_CALLS)

    if label == "DROP":
        dropped += 1
        continue

    # KEEP or UNCLEAR
    if label == "KEEP":
        kept += 1
    else:
        unclear += 1

    all_data.append({"id": current_id, "text": text})
    current_id += 1

    if len(all_data) % 100 == 0:
        print(f"  Progress: {len(all_data)}/{TOTAL_NEEDED} (KEEP={kept}, UNCLEAR={unclear}, DROP={dropped})")

if len(all_data) < TOTAL_NEEDED:
    print(f"WARNING: Only collected {len(all_data)} comments. Pool exhausted.")

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

print(f"DONE. KEEP={kept}, UNCLEAR={unclear}, DROP={dropped}")
