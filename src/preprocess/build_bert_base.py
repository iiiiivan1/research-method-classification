# -*- coding: utf-8 -*-

import os

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# =========================
# Configuration
# =========================

# Root directory of the raw TXT files.
DATA_ROOT = Path("process_Data")

# The manifest file stores the fixed train / validation / test split.
THIS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = THIS_DIR / "split_manifest.json"

MAX_TITLE_CHARS = 800
MAX_ABS_CHARS = 4000

OUT_DIR = Path("./out_bert_multilabel_manifest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train_bert.jsonl"
VAL_OUT = OUT_DIR / "val_bert.jsonl"
TEST_OUT = OUT_DIR / "test_bert.jsonl"

# Output mode:
# - "pair": store title and abstract as separate fields
# - "text": concatenate title and abstract into a single text field
OUTPUT_MODE = "pair"


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

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
LABEL_SET = set(LABELS)


# =========================
# Regular Expressions
# =========================

TITLE_RE = re.compile(r"<\s*title\b[^>]*>(.*?)<\s*/\s*title\s*>", re.I | re.S)
ABS_RE = re.compile(r"<\s*abstract\b[^>]*>(.*?)<\s*/\s*abstract\s*>", re.I | re.S)
METHOD_RE = re.compile(r"<\s*method\b[^>]*>(.*?)<\s*/\s*method\s*>", re.I | re.S)

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def extract(text: str, regex: re.Pattern) -> str:
    """Extract text content matched by a regular expression."""
    match = regex.search(text)
    return match.group(1).strip() if match else ""


def clean_text(text: str, max_chars: int = 0) -> str:
    """Remove XML-like tags, normalize whitespace, and optionally truncate text."""
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()

    return text


def parse_methods(raw: str) -> List[str]:
    """
    Parse method labels from the <method> field.

    Labels outside the predefined label set are ignored.
    If no valid label remains, the sample is assigned to 'Other'.
    """
    items = [item.strip() for item in raw.split(";") if item.strip()]
    valid_labels = []

    for item in items:
        if item in LABEL_SET and item != "Other":
            valid_labels.append(item)

    valid_labels = sorted(set(valid_labels), key=lambda x: LABELS.index(x))

    if not valid_labels:
        return ["Other"]

    return valid_labels


def labels_to_multihot(label_names: List[str]) -> List[int]:
    """Convert label names into a multi-hot vector."""
    y = [0] * len(LABELS)

    for label in label_names:
        y[LABEL_TO_ID[label]] = 1

    return y


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> Dict:
    """Load a split manifest containing train / validation / test file names."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_keys = ["train_files", "val_files", "test_files"]

    for key in required_keys:
        if key not in manifest:
            raise KeyError(f"Missing key in manifest: {key}")

    return manifest


def check_split_overlap(
    train_names: List[str],
    val_names: List[str],
    test_names: List[str],
) -> None:
    """Ensure that the train / validation / test splits do not overlap."""
    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            "Manifest split overlap detected:\n"
            f"train∩val={len(train_val_overlap)}, "
            f"train∩test={len(train_test_overlap)}, "
            f"val∩test={len(val_test_overlap)}"
        )


def build_file_map(file_list: List[Path]) -> Dict[str, Path]:
    """
    Build a mapping from basename to file path.

    If duplicate basenames are found, an error is raised to avoid incorrect
    matching between manifest entries and raw TXT files.
    """
    file_map = {}
    duplicates = []

    for path in file_list:
        name = path.name

        if name in file_map:
            duplicates.append(name)
        else:
            file_map[name] = path

    if duplicates:
        duplicate_examples = "\n".join(sorted(set(duplicates))[:20])
        raise ValueError(
            "Duplicate basenames found under DATA_ROOT; "
            "cannot safely match manifest entries by filename.\n"
            f"Examples:\n{duplicate_examples}"
        )

    return file_map


def build_record_from_file(path: Path) -> Optional[Dict]:
    """Build one BERT-style multi-label classification record from a raw TXT file."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    title_raw = extract(text, TITLE_RE)
    abstract_raw = extract(text, ABS_RE)
    method_raw = extract(text, METHOD_RE)

    title = clean_text(title_raw, MAX_TITLE_CHARS)
    abstract = clean_text(abstract_raw, MAX_ABS_CHARS)

    if not title or not abstract or not method_raw:
        return None

    label_names = parse_methods(method_raw)
    labels = labels_to_multihot(label_names)

    if OUTPUT_MODE == "pair":
        return {
            "title": title,
            "abstract": abstract,
            "labels": labels,
            "label_names": label_names,
            "source_file": path.name,
        }

    if OUTPUT_MODE == "text":
        return {
            "text": f"{title} [SEP] {abstract}",
            "labels": labels,
            "label_names": label_names,
            "source_file": path.name,
        }

    raise ValueError(f"Unknown OUTPUT_MODE: {OUTPUT_MODE}")


def collect_split_records(
    split_names: List[str],
    file_map: Dict[str, Path],
    split_name: str,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Collect records for one split.

    Returns:
        records: successfully parsed samples
        missing_files: files listed in the manifest but not found under DATA_ROOT
        skipped_invalid: files found but skipped due to missing required fields
    """
    records = []
    missing_files = []
    skipped_invalid = []

    for name in split_names:
        path = file_map.get(name)

        if path is None:
            missing_files.append(name)
            continue

        record = build_record_from_file(path)

        if record is None:
            skipped_invalid.append(name)
            continue

        record["split"] = split_name
        records.append(record)

    return records, missing_files, skipped_invalid


def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    manifest = load_manifest(MANIFEST_PATH)

    file_list = list(DATA_ROOT.rglob("*.txt"))
    print(f"Found {len(file_list)} TXT files in {DATA_ROOT}")

    file_map = build_file_map(file_list)

    train_names = manifest["train_files"]
    val_names = manifest["val_files"]
    test_names = manifest["test_files"]

    check_split_overlap(train_names, val_names, test_names)

    train, train_missing, train_skipped = collect_split_records(
        train_names,
        file_map,
        "train",
    )
    val, val_missing, val_skipped = collect_split_records(
        val_names,
        file_map,
        "val",
    )
    test, test_missing, test_skipped = collect_split_records(
        test_names,
        file_map,
        "test",
    )

    all_missing = train_missing + val_missing + test_missing
    all_skipped = train_skipped + val_skipped + test_skipped

    if not train and not val and not test:
        raise RuntimeError(
            "No valid samples were extracted from manifest splits. "
            "Please check DATA_ROOT, TXT format, and the manifest file."
        )

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    write_jsonl(TEST_OUT, test)

    print("-" * 60)
    print("Done with manifest-based split.")
    print(f"Mode: {OUTPUT_MODE}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Train target / actual: {len(train_names)} / {len(train)}")
    print(f"Validation target / actual: {len(val_names)} / {len(val)}")
    print(f"Test target / actual: {len(test_names)} / {len(test)}")
    print(f"Saved: {TRAIN_OUT}")
    print(f"Saved: {VAL_OUT}")
    print(f"Saved: {TEST_OUT}")
    print(f"Missing files from manifest: {len(all_missing)}")
    print(f"Skipped invalid files: {len(all_skipped)}")
    print("-" * 60)

    if all_missing:
        missing_path = OUT_DIR / "missing_from_manifest.txt"
        missing_path.write_text("\n".join(all_missing), encoding="utf-8")
        print(f"Missing file list saved to: {missing_path}")

    if all_skipped:
        skipped_path = OUT_DIR / "skipped_invalid.txt"
        skipped_path.write_text("\n".join(all_skipped), encoding="utf-8")
        print(f"Skipped-invalid file list saved to: {skipped_path}")


if __name__ == "__main__":
    main()