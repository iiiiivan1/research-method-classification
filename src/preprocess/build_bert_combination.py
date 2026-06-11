# -*- coding: utf-8 -*-

import os

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# =========================
# Configuration
# =========================

# Directory containing the raw TXT corpus.
DATA_ROOT = Path("m")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Number of physical-position segments.
N_PARTS = 10

# Boundary adjustment parameters.
# WINDOW_CHARS controls the local search window around each theoretical cut point.
# MIN_SEG_CHARS prevents extremely short segments.
WINDOW_CHARS = 240
MIN_SEG_CHARS = 200

# Output directory for P4 datasets.
# P4 contains all pairwise combinations of two body segments with Title + Abstract.
OUT_DIR = Path("./out_bert_p4_pairs_ta")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_MANIFEST = OUT_DIR / "split_manifest.json"

# Reuse the split manifest generated for P3 experiments.
# This guarantees identical train/validation/test partitions across P3 and P4 datasets.
# Place the P3 split manifest in the current working directory and name it:
# p3_split_manifest.json
REUSE_MANIFEST = Path("./p3_split_manifest.json")

# Optional truncation limits for extremely long titles and abstracts.
TITLE_MAX_CHARS = 300
ABS_MAX_CHARS = 1500


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
FULLTEXT_RE = re.compile(r"<\s*fulltext\b[^>]*>(.*?)<\s*/\s*fulltext\s*>", re.I | re.S)

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# Punctuation marks used to identify sentence boundaries.
PUNCT = set(".!?;。！？；")


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


def clean_fulltext_physical(text: str) -> str:
    """
    Clean full-text content while preserving the original document order.

    Unlike title and abstract processing, no length truncation is applied
    because the full text will later be segmented by physical position.
    """
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def parse_methods(raw: str) -> List[str]:
    """
    Parse research method labels from the <method> field.

    Invalid labels are ignored.
    If no valid label remains, the sample is assigned to 'Other'.
    """
    items = [item.strip() for item in raw.split(";") if item.strip()]
    valid_labels = []

    for item in items:
        if item in LABEL_SET and item != "Other":
            valid_labels.append(item)

    valid_labels = sorted(set(valid_labels), key=lambda x: LABELS.index(x))

    return valid_labels if valid_labels else ["Other"]


def labels_to_multihot(label_names: List[str]) -> List[int]:
    """Convert label names into a multi-hot representation."""
    y = [0] * len(LABELS)

    for label in label_names:
        y[LABEL_TO_ID[label]] = 1

    return y


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Physical Position Segmentation
# =========================

def _boundary_score(text: str, cut: int) -> int:
    """
    Score a candidate cut point.

    A higher score is assigned to cut points that align with sentence-ending
    punctuation followed by whitespace. Lower but positive scores are assigned
    to whitespace boundaries.
    """
    n = len(text)

    if cut <= 0 or cut >= n:
        return 0

    if text[cut].isspace():
        previous_char = text[cut - 1]
        return 2 if previous_char in PUNCT else 1

    if text[cut - 1].isspace():
        previous_previous_char = text[cut - 2] if cut - 2 >= 0 else ""
        return 2 if previous_previous_char in PUNCT else 1

    return 0


def _pick_cut_near(
    text: str,
    ideal: int,
    left_limit: int,
    right_limit: int,
    window: int,
) -> int:
    """
    Select a cut point near the theoretical boundary.

    The function searches within a local window and prefers boundaries that
    align with sentence-ending punctuation or whitespace.
    """
    lower_bound = max(left_limit, ideal - window)
    upper_bound = min(right_limit, ideal + window)

    best_cut = None
    best_score = -1
    best_distance = 10**18

    for distance in range(0, upper_bound - lower_bound + 1):
        for cut in (ideal - distance, ideal + distance):
            if cut < lower_bound or cut > upper_bound:
                continue

            score = _boundary_score(text, cut)

            if score <= 0:
                continue

            if score > best_score or (score == best_score and distance < best_distance):
                best_score = score
                best_distance = distance
                best_cut = cut

        if best_score == 2 and best_distance <= 3:
            break

    if best_cut is None:
        best_cut = max(left_limit, min(right_limit, ideal))

    return best_cut


def split_chars_into_parts_b(
    text: str,
    n_parts: int,
    window: int,
    min_seg_chars: int,
) -> List[Tuple[int, int]]:
    """
    Divide a document into N physical-position segments.

    The document is first divided using theoretical percentage-based boundaries
    such as 10%, 20%, ..., 90%.

    To avoid splitting inside words or sentences, each boundary is adjusted
    within a local search window. Preference is given to:

    1. Sentence-ending punctuation
    2. Whitespace boundaries

    A minimum segment length constraint is applied to prevent extremely short
    segments.

    Returns:
        A list of (start, end) character offsets.
    """
    n_chars = len(text)

    if n_chars == 0:
        return [(0, 0)] * n_parts

    if n_chars < n_parts * 10:
        spans = []

        for i in range(n_parts):
            start = (i * n_chars) // n_parts
            end = ((i + 1) * n_chars) // n_parts

            if i == n_parts - 1:
                end = n_chars

            spans.append((start, end))

        return spans

    cuts = [0]
    previous_cut = 0

    for i in range(1, n_parts):
        ideal_raw = (i * n_chars) // n_parts
        ideal_for_cut = ideal_raw

        remaining_parts = n_parts - i
        right_limit = n_chars - remaining_parts
        left_limit = previous_cut + 1

        if ideal_for_cut - previous_cut < min_seg_chars:
            ideal_for_cut = min(right_limit, previous_cut + min_seg_chars)

        ideal_span_raw = max(1, ideal_raw - previous_cut)
        dynamic_window = min(window, max(1, int(0.2 * ideal_span_raw)))

        cut = _pick_cut_near(
            text=text,
            ideal=ideal_for_cut,
            left_limit=left_limit,
            right_limit=right_limit,
            window=dynamic_window,
        )

        if cut <= previous_cut:
            cut = min(right_limit, previous_cut + 1)

        cuts.append(cut)
        previous_cut = cut

    cuts.append(n_chars)

    return [(cuts[i], cuts[i + 1]) for i in range(n_parts)]


def load_manifest(path: Path) -> Dict:
    """Load a JSON split manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(path: Path, payload: Dict) -> None:
    """Save a JSON split manifest."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_row(doc: Dict, i: int, j: int) -> Dict:
    """
    Construct one P4 sample.

    P4 consists of:

        Title
        Abstract
        Segment_i
        Segment_j

    where Segment_i and Segment_j correspond to two physical-position segments
    extracted from the full text.

    The resulting sample is used for multi-label research method classification.
    """
    start_i, end_i = doc["spans"][i]
    start_j, end_j = doc["spans"][j]

    chunk_i = doc["fulltext"][start_i:end_i].strip()
    chunk_j = doc["fulltext"][start_j:end_j].strip()

    parts = [
        f"Title: {doc['title']}" if doc["title"] else "Title:",
        f"Abstract: {doc['abstract']}" if doc["abstract"] else "Abstract:",
        chunk_i,
        chunk_j,
    ]

    text = "\n".join(parts[:2]) + "\n\n" + parts[2] + "\n\n" + parts[3]

    return {
        "title": doc["title"],
        "abstract": doc["abstract"],
        "chunk_i": chunk_i,
        "chunk_j": chunk_j,
        "text": text,
        "labels": doc["labels"],
        "label_names": doc["label_names"],
        "source_file": doc["file"],
        "pair": f"{i:02d}-{j:02d}",
    }


def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    files = sorted(DATA_ROOT.rglob("*.txt"))
    print(f"Found {len(files)} TXT files in {DATA_ROOT}")

    docs = []
    skipped = 0
    skipped_no_fulltext = 0
    skipped_no_method = 0

    # Read and parse raw TXT files.
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")

        method_raw = extract(text, METHOD_RE)
        fulltext_raw = extract(text, FULLTEXT_RE)
        fulltext = clean_fulltext_physical(fulltext_raw)

        if not method_raw:
            skipped += 1
            skipped_no_method += 1
            continue

        if not fulltext:
            skipped += 1
            skipped_no_fulltext += 1
            continue

        # Missing title or abstract does not cause the sample to be skipped.
        title_raw = extract(text, TITLE_RE)
        abstract_raw = extract(text, ABS_RE)

        title = clean_text(title_raw, max_chars=TITLE_MAX_CHARS) if title_raw else ""
        abstract = clean_text(abstract_raw, max_chars=ABS_MAX_CHARS) if abstract_raw else ""

        label_names = parse_methods(method_raw)
        labels = labels_to_multihot(label_names)

        spans = split_chars_into_parts_b(
            text=fulltext,
            n_parts=N_PARTS,
            window=WINDOW_CHARS,
            min_seg_chars=MIN_SEG_CHARS,
        )

        docs.append(
            {
                "file": path.name,
                "title": title,
                "abstract": abstract,
                "labels": labels,
                "label_names": label_names,
                "fulltext": fulltext,
                "spans": spans,
            }
        )

    if not docs:
        raise RuntimeError(
            "No valid samples were extracted. "
            "Please check whether <method> and <fulltext> exist in the TXT files."
        )

    print(
        f"Valid documents={len(docs)} | "
        f"Skipped={skipped} "
        f"(missing_method={skipped_no_method}, "
        f"missing_fulltext={skipped_no_fulltext})"
    )

    # Reuse the P3 split manifest to ensure comparable splits.
    if not REUSE_MANIFEST.exists():
        raise FileNotFoundError(
            f"REUSE_MANIFEST not found: {REUSE_MANIFEST}\n"
            "Please place the P3 split_manifest.json in the working directory "
            "and rename it to p3_split_manifest.json, or update REUSE_MANIFEST."
        )

    manifest = load_manifest(REUSE_MANIFEST)
    train_files = manifest["train_files"]
    val_files = manifest["val_files"]
    test_files = manifest["test_files"]

    by_file = {doc["file"]: doc for doc in docs}

    missing = [
        filename
        for filename in (train_files + val_files + test_files)
        if filename not in by_file
    ]

    if missing:
        raise RuntimeError(
            f"Manifest contains {len(missing)} files that are missing "
            f"from the current parsed documents. Example missing files: {missing[:10]}"
        )

    train_docs = [by_file[filename] for filename in train_files]
    val_docs = [by_file[filename] for filename in val_files]
    test_docs = [by_file[filename] for filename in test_files]

    # Save a copy of the reused manifest for traceability.
    save_manifest(SPLIT_MANIFEST, manifest)

    print(f"Reused split manifest from: {REUSE_MANIFEST}")
    print(f"Saved manifest copy to: {SPLIT_MANIFEST}")

    # Record segment character-length statistics.
    seg_len_map = defaultdict(list)
    seg_empty_map = defaultdict(int)

    pairs = [(i, j) for i in range(N_PARTS) for j in range(i + 1, N_PARTS)]

    print(
        f"Generating {len(pairs)} P4 segment combinations "
        f"(all pairwise combinations of {N_PARTS} segments)."
    )

    for i, j in pairs:
        pair_name = f"{i:02d}-{j:02d}"

        train_rows = [build_row(doc, i, j) for doc in train_docs]
        val_rows = [build_row(doc, i, j) for doc in val_docs]
        test_rows = [build_row(doc, i, j) for doc in test_docs]

        for row in train_rows + val_rows + test_rows:
            chunk_i_len = len(row["chunk_i"])
            chunk_j_len = len(row["chunk_j"])

            seg_len_map[i].append(chunk_i_len)
            seg_len_map[j].append(chunk_j_len)

            if chunk_i_len == 0:
                seg_empty_map[i] += 1
            if chunk_j_len == 0:
                seg_empty_map[j] += 1

        write_jsonl(OUT_DIR / f"train_{pair_name}.jsonl", train_rows)
        write_jsonl(OUT_DIR / f"val_{pair_name}.jsonl", val_rows)
        write_jsonl(OUT_DIR / f"test_{pair_name}.jsonl", test_rows)

        print(
            f"[pair {pair_name}] "
            f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    print("\n===== Segment Character-Length Statistics =====")

    for seg_id in range(N_PARTS):
        lengths = seg_len_map[seg_id]

        if not lengths:
            print(f"seg {seg_id:02d}: n=0")
            continue

        mean_length = sum(lengths) / len(lengths)
        median_length = int(statistics.median(lengths))
        max_length = max(lengths)

        print(
            f"seg {seg_id:02d}: "
            f"n={len(lengths)} mean={mean_length:.1f} "
            f"median={median_length} max={max_length} "
            f"empty={seg_empty_map[seg_id]}"
        )

    print("\nDone.")
    print(f"Outputs saved to: {OUT_DIR.resolve()}")
    print(f"Split manifest saved to: {SPLIT_MANIFEST.resolve()}")


if __name__ == "__main__":
    main()