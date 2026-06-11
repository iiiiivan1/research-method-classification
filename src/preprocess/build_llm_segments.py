# -*- coding: utf-8 -*-

import os
import re
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from transformers import AutoTokenizer


# =========================
# Environment Settings
# =========================

# Do NOT hard-code HuggingFace tokens in the source code.
# If needed, set the token in the terminal:
# export HF_TOKEN="your_huggingface_token"
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Optional HuggingFace mirror endpoint.
# You can set it in the terminal if needed:
# export HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


# =========================
# Runtime Mode
# =========================

# Choose one runtime mode: "qwen", "deepseek", or "llama".
RUN = "qwen"


# =========================
# Runtime Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RuntimeCfg:
    """Runtime configuration for different LLM backbones."""

    run: str
    model_dir: str
    adapter_dir: str
    merge_system_to_user: bool
    tokenizer_kwargs: Dict[str, Any]
    local_files_only: bool


MODEL_REGISTRY = {
    "qwen": RuntimeCfg(
        run="qwen",
        model_dir="/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        adapter_dir=str(BASE_DIR / "outputs/chat-sft-qlora/adapter"),
        merge_system_to_user=True,
        tokenizer_kwargs=dict(use_fast=True, trust_remote_code=True),
        local_files_only=True,
    ),
    "deepseek": RuntimeCfg(
        run="deepseek",
        model_dir="/root/autodl-tmp/model_cache/deepseek-ai/deepseek-llm-7b-chat",
        adapter_dir=str(BASE_DIR / "outputs/deepseek/base1"),
        merge_system_to_user=True,
        tokenizer_kwargs=dict(use_fast=True, trust_remote_code=True),
        local_files_only=True,
    ),
    "llama": RuntimeCfg(
        run="llama",
        model_dir=str(BASE_DIR / "model_cache/LLM-Research/Meta-Llama-3-8B-Instruct"),
        adapter_dir=str(BASE_DIR / "outputs/llama/base/adapter"),
        merge_system_to_user=False,
        tokenizer_kwargs=dict(use_fast=True, trust_remote_code=False),
        local_files_only=True,
    ),
}


def get_runtime(run: str) -> RuntimeCfg:
    """Return the runtime configuration for the selected model."""
    run = run.strip().lower()

    if run not in MODEL_REGISTRY:
        raise ValueError(
            f"RUN must be one of {list(MODEL_REGISTRY.keys())}, got: {run}"
        )

    return MODEL_REGISTRY[run]


def load_tokenizer(cfg: RuntimeCfg):
    """Load the tokenizer for the selected model."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_dir,
        local_files_only=cfg.local_files_only,
        **cfg.tokenizer_kwargs,
    )

    # Some LLM tokenizers do not define a pad token.
    # Use the EOS token as a fallback to avoid batching errors.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def merge_system_to_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge the system message into the first user message.

    This is useful for models or chat templates that do not support
    the system role properly.
    """
    system_content = "\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "system"
    ).strip()

    remaining_messages = [
        message for message in messages if message.get("role") != "system"
    ]

    if not system_content or not remaining_messages:
        return messages

    if remaining_messages[0].get("role") == "user":
        remaining_messages[0]["content"] = (
            system_content + "\n\n" + remaining_messages[0]["content"]
        )
        return remaining_messages

    return [{"role": "user", "content": system_content}] + remaining_messages


# =========================
# Data and Output Settings
# =========================

# Raw TXT files should be placed under this directory.
DATA_ROOT = BASE_DIR / "m"

# Output directory for segmented LLM training data.
OUT_DIR = BASE_DIR / f"out_token10_{RUN}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

MAX_TITLE_CHARS = 600
MAX_ABS_CHARS = 3000

# Number of physical-position segments.
N_PARTS = 10

# Maximum number of tokens per segment.
# Set to 0 to disable truncation.
MAX_SEG_TOKENS = 0


# =========================
# Labels and Regular Expressions
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

TITLE_RE = re.compile(r"<\s*title\b[^>]*>(.*?)<\s*/\s*title\s*>", re.I | re.S)
ABS_RE = re.compile(r"<\s*abstract\b[^>]*>(.*?)<\s*/\s*abstract\s*>", re.I | re.S)
METHOD_RE = re.compile(r"<\s*method\b[^>]*>(.*?)<\s*/\s*method\s*>", re.I | re.S)
FULLTEXT_RE = re.compile(r"<\s*fulltext\b[^>]*>(.*?)<\s*/\s*fulltext\s*>", re.I | re.S)

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

    return valid_labels if valid_labels else ["Other"]


def split_tokens_into_parts(token_ids: List[int], n_parts: int) -> List[Tuple[int, int]]:
    """
    Split a token sequence into equal physical-position parts.

    Returns a list of (start, end) token index spans covering [0, len(token_ids)].
    """
    n_tokens = len(token_ids)
    spans = []

    for i in range(n_parts):
        start = (i * n_tokens) // n_parts
        end = ((i + 1) * n_tokens) // n_parts

        if i == n_parts - 1:
            end = n_tokens

        spans.append((start, end))

    return spans


def seg_range_label(seg_id: int, n_parts: int) -> str:
    """Return the percentage range label for a segment."""
    lower = int(100 * seg_id / n_parts)
    upper = int(100 * (seg_id + 1) / n_parts)

    return f"{lower}-{upper}%"


def make_sample(
    title: str,
    abstract: str,
    seg_text: str,
    seg_id: int,
    labels: List[str],
    file_id: str,
    cfg: RuntimeCfg,
) -> Dict[str, Any]:
    """Build one chat-style supervised fine-tuning sample."""
    labels_str = "; ".join(LABELS)

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
        f"{labels_str}\n\n"
        "### Input Data\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Body Chunk ({seg_range_label(seg_id, N_PARTS)}, seg_id={seg_id}):\n"
        f"{seg_text}\n\n"
        "### Output\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "; ".join(labels)},
    ]

    if cfg.merge_system_to_user:
        messages = merge_system_to_user(messages)

    return {
        "id": file_id,
        "seg_id": seg_id,
        "messages": messages,
    }


def write_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write JSON objects to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for obj in data:
            file.write(json.dumps(obj, ensure_ascii=False) + "\n")


def percentile(values: List[int], p: int) -> int:
    """Calculate the p-th percentile from a list of integer values."""
    if not values:
        return 0

    values = sorted(values)
    index = int(math.ceil((p / 100) * len(values))) - 1
    index = max(0, min(index, len(values) - 1))

    return int(values[index])


def main() -> None:
    cfg = get_runtime(RUN)

    print("RUN =", cfg.run)
    print("MODEL_DIR =", cfg.model_dir)
    print("ADAPTER_DIR =", cfg.adapter_dir)
    print("MERGE_SYSTEM_TO_USER =", cfg.merge_system_to_user)
    print("DATA_ROOT =", DATA_ROOT)
    print("OUT_DIR =", OUT_DIR)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    tokenizer = load_tokenizer(cfg)

    files = sorted(DATA_ROOT.rglob("*.txt"))
    print(f"Found {len(files)} TXT files.")

    # Record the token length distribution for each 10% segment.
    seg_len_map = defaultdict(list)
    seg_empty_map = defaultdict(int)

    # Load each document once and tokenize the full text once.
    docs = []
    skipped = 0

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")

        title = clean_text(extract(text, TITLE_RE), MAX_TITLE_CHARS)
        abstract = clean_text(extract(text, ABS_RE), MAX_ABS_CHARS)
        method_raw = extract(text, METHOD_RE)
        fulltext = clean_text(extract(text, FULLTEXT_RE), 0)

        if not title or not abstract or not method_raw or not fulltext:
            skipped += 1
            continue

        labels = parse_methods(method_raw)

        # Tokenize the full text only once for each document.
        token_ids = tokenizer.encode(fulltext, add_special_tokens=False)

        if len(token_ids) == 0:
            skipped += 1
            continue

        docs.append(
            {
                "file": path.name,
                "title": title,
                "abstract": abstract,
                "labels": labels,
                "token_ids": token_ids,
            }
        )

    if not docs:
        raise RuntimeError(
            "No valid samples were extracted. "
            "Please check whether <title>, <abstract>, <fulltext>, and <method> exist."
        )

    print(f"Valid documents = {len(docs)}, skipped = {skipped}")

    # Use a fixed split shared by all segment files.
    random.seed(RANDOM_SEED)
    random.shuffle(docs)

    n_docs = len(docs)
    n_train = int(n_docs * TRAIN_RATIO)
    n_val = int(n_docs * VAL_RATIO)

    train_docs = docs[:n_train]
    val_docs = docs[n_train:n_train + n_val]
    test_docs = docs[n_train + n_val:]

    # Generate ten segmented datasets: seg0 to seg9.
    for seg_id in range(N_PARTS):
        seg_train = []
        seg_val = []
        seg_test = []

        for split_docs, bucket in [
            (train_docs, seg_train),
            (val_docs, seg_val),
            (test_docs, seg_test),
        ]:
            for doc in split_docs:
                token_ids = doc["token_ids"]
                spans = split_tokens_into_parts(token_ids, N_PARTS)
                start, end = spans[seg_id]
                segment_token_ids = token_ids[start:end]

                segment_length = len(segment_token_ids)

                if segment_length == 0:
                    seg_empty_map[seg_id] += 1
                else:
                    seg_len_map[seg_id].append(segment_length)

                if MAX_SEG_TOKENS and len(segment_token_ids) > MAX_SEG_TOKENS:
                    segment_token_ids = segment_token_ids[:MAX_SEG_TOKENS]

                segment_text = tokenizer.decode(
                    segment_token_ids,
                    skip_special_tokens=True,
                ).strip()

                bucket.append(
                    make_sample(
                        title=doc["title"],
                        abstract=doc["abstract"],
                        seg_text=segment_text,
                        seg_id=seg_id,
                        labels=doc["labels"],
                        file_id=doc["file"],
                        cfg=cfg,
                    )
                )

        write_jsonl(OUT_DIR / f"train_tok{seg_id:02d}.jsonl", seg_train)
        write_jsonl(OUT_DIR / f"val_tok{seg_id:02d}.jsonl", seg_val)
        write_jsonl(OUT_DIR / f"test_tok{seg_id:02d}.jsonl", seg_test)

        print(
            f"[seg {seg_id:02d} {seg_range_label(seg_id, N_PARTS)}] "
            f"train={len(seg_train)} val={len(seg_val)} test={len(seg_test)}"
        )

    # Save the split manifest for reproducibility.
    with (OUT_DIR / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "run": cfg.run,
                "model_dir": cfg.model_dir,
                "adapter_dir": cfg.adapter_dir,
                "merge_system_to_user": cfg.merge_system_to_user,
                "train_files": [doc["file"] for doc in train_docs],
                "val_files": [doc["file"] for doc in val_docs],
                "test_files": [doc["file"] for doc in test_docs],
                "n_parts": N_PARTS,
                "max_seg_tokens": MAX_SEG_TOKENS,
                "seed": RANDOM_SEED,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Print token statistics for each 10% segment before truncation.
    print("\n===== Token Statistics for 10% Segments =====")
    print(f"Number of documents used for statistics: {len(docs)}")

    for seg_id in range(N_PARTS):
        lengths = seg_len_map[seg_id]
        empty_count = seg_empty_map[seg_id]

        if not lengths:
            print(
                f"seg {seg_id:02d} ({seg_range_label(seg_id, N_PARTS)}) | "
                f"n=0 empty={empty_count}"
            )
            continue

        mean_length = sum(lengths) / len(lengths)
        median_length = int(statistics.median(lengths))
        p95_length = percentile(lengths, 95)
        max_length = max(lengths)

        print(
            f"seg {seg_id:02d} ({seg_range_label(seg_id, N_PARTS)}) | "
            f"n={len(lengths)} mean={mean_length:.2f} "
            f"median={median_length} p95={p95_length} "
            f"max={max_length} | empty={empty_count}"
        )

    print("\nDone. Outputs saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()