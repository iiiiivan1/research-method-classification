# -*- coding: utf-8 -*-

import re
import json
import random
from pathlib import Path


# =========================
# Configuration
# =========================

# Root directory of the raw TXT files.
# Each TXT file is expected to contain fields such as <title>, <abstract>, and <method>.
DATA_ROOT = Path("data_root data/raw --out_dir data/processed")

# Train / validation / test split ratio.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Maximum character lengths for title and abstract.
# The baseline setting only uses Title + Abstract, so full-text truncation is not needed.
MAX_TITLE_CHARS = 600
MAX_ABS_CHARS = 3000

# Output file names for the Title + Abstract baseline.
TRAIN_OUT = "train_base.jsonl"
VAL_OUT = "val_base.jsonl"
TEST_OUT = "test_base.jsonl"


# =========================
# Label Set
# =========================

LABELS = [
    "Bibliometrics",
    "Content Analysis",
    "Delphi Study",
    "Ethnography / Field Study",
    "Experiment",
    "Focus Group",
    "Historical Method",
    "Interview",
    "Observation",
    "Questionnaire",
    "Research Diary / Journal",
    "Theoretical Approach",
    "Think Aloud Protocol",
    "Transaction Log Analysis",
    "Webometrics",
    "Other",
]

LABEL_SET = set(LABELS)


# =========================
# Regular Expressions
# =========================

TITLE_RE = re.compile(r"<\s*title\b[^>]*>(.*?)<\s*/\s*title\s*>", re.I | re.S)
ABS_RE = re.compile(r"<\s*abstract\b[^>]*>(.*?)<\s*/\s*abstract\s*>", re.I | re.S)
METHOD_RE = re.compile(r"<\s*method\b[^>]*>(.*?)<\s*/\s*method\s*>", re.I | re.S)

# Remove XML-like tags.
TAG_RE = re.compile(r"<[^>]+>")

# Normalize repeated whitespace.
WS_RE = re.compile(r"\s+")


def extract(txt: str, regex: re.Pattern) -> str:
    """Extract the content matched by a regular expression."""
    match = regex.search(txt)
    return match.group(1).strip() if match else ""


def clean_text(text: str, max_chars: int = 0) -> str:
    """Remove tags, normalize whitespace, and optionally truncate text."""
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()

    return text


def parse_methods(raw: str):
    """
    Parse method labels from the <method> field.

    Labels that are not included in the predefined label set are mapped to 'Other'.
    """
    items = [x.strip() for x in raw.split(";") if x.strip()]
    labels = []

    for item in items:
        item_clean = item.strip()

        if item_clean in LABEL_SET:
            labels.append(item_clean)
        else:
            if "Other" not in labels:
                labels.append("Other")

    labels = sorted(
        set(labels),
        key=lambda x: LABELS.index(x) if x in LABELS else 999
    )

    return labels


def make_sample(title: str, abstract: str, labels):
    """
    Build one chat-style supervised fine-tuning sample.

    The input contains only the title and abstract, corresponding to the TA baseline.
    """
    system_prompt = (
        "You are an expert academic research method classifier. "
        "Your task is to identify the research methods used in the paper based on the text provided."
    )

    user_prompt = (
        "### Instructions\n"
        "1. Analyze the content below.\n"
        "2. Select methods ONLY from the Candidate List.\n"
        "3. Output the selected labels separated by semicolons (;).\n"
        "4. Do NOT output numbers, explanations, or any extra text.\n"
        "5. If no method is detected, output 'Other'.\n\n"
        "### Candidate List\n"
        f"{'; '.join(LABELS)}\n\n"
        "### Input Data\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        "### Output\n"
    )

    assistant_output = "; ".join(labels)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
    }


def write_jsonl(path: str, data):
    """Write a list of JSON objects to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def guess_reason(txt: str):
    """Guess why a raw TXT file cannot be parsed correctly."""
    if "<title>" not in txt:
        return "missing <title>"
    if "<abstract>" not in txt:
        return "missing <abstract>"
    if "<method>" not in txt:
        return "missing <method>"
    return "unknown format error"


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    records = []
    skipped_missing = 0
    skipped_no_label = 0
    skipped_details = []

    file_list = list(DATA_ROOT.rglob("*.txt"))
    print(f"Found {len(file_list)} TXT files. Start processing...")

    for path in file_list:
        txt = path.read_text(encoding="utf-8", errors="ignore")

        # Step 1: Extract basic fields.
        title_raw = extract(txt, TITLE_RE)
        abstract_raw = extract(txt, ABS_RE)
        method_raw = extract(txt, METHOD_RE)

        # Step 2: Clean and truncate title and abstract.
        title_clean = clean_text(title_raw, MAX_TITLE_CHARS)
        abstract_clean = clean_text(abstract_raw, MAX_ABS_CHARS)

        # Step 3: Check missing fields.
        missing = []
        if not title_clean:
            missing.append("title")
        if not abstract_clean:
            missing.append("abstract")
        if not method_raw:
            missing.append("method")

        if missing:
            skipped_missing += 1
            skipped_details.append(
                {
                    "file": path.name,
                    "missing": ";".join(missing),
                    "reason": guess_reason(txt),
                }
            )
            continue

        # Step 4: Parse method labels.
        labels = parse_methods(method_raw)
        if not labels:
            skipped_no_label += 1
            continue

        # Step 5: Build one training sample.
        sample = make_sample(title_clean, abstract_clean, labels)
        records.append(sample)

    if not records:
        raise RuntimeError("No valid samples were extracted.")

    # Step 6: Split the dataset into train / validation / test sets.
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    total = len(records)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    write_jsonl(TEST_OUT, test)

    print("-" * 30)
    print(f"Processing finished. Total samples: {total}")
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"Skipped due to missing fields: {skipped_missing}")
    print(f"Skipped due to empty labels: {skipped_no_label}")
    print(f"Generated files: {TRAIN_OUT}, {VAL_OUT}, {TEST_OUT}")
    print("-" * 30)

    if skipped_details:
        print("First 5 skipped samples:")
        for item in skipped_details[:5]:
            print(item)


if __name__ == "__main__":
    main()