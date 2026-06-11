# -*- coding: utf-8 -*-
"""
evaluate_llm.py
---------------

Evaluation script for LLM-based pair + TA experiments.

This script supports two modes:

1. Generation mode:
   Generate method-label predictions for validation and test files.

2. Evaluation mode:
   Evaluate generated label strings against gold labels.

The input setting follows the pair + TA format:
Title + Abstract + two body chunks.

The input truncation strategy is aligned with the training script:
only the two body chunk contents are truncated, while the title,
abstract, prompt instructions, and chunk headers are preserved.
"""

import os
import re
import json
import gc
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
from typing import Optional, List, Tuple, Dict

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from metrics import (
    LABELS,
    pred_gold_sets,
    compute_set_metrics_with_report,
    print_classification_report,
)


# =========================
# Configuration
# =========================

RUN = "qwen"          # "qwen", "deepseek", or "llama"
MODE = "eval"        # "gen" or "eval"

# LLM pair + TA data directory.
# Expected files:
#   val0-1.jsonl
#   test0-1.jsonl
DATA_DIR = Path(f"out_10ta_{RUN}")

# Adapter directory.
# Expected structure:
#   outputs_pairta/qwen/pair0-1/adapter
ADAPTER_ROOT = Path("outputs_pairta") / RUN

# Pair range to evaluate.
START_PAIR = "0-1"
END_PAIR = "8-9"

# Generation settings.
MAX_NEW_TOKENS = 96

# Maximum input length for prompt truncation.
# This controls input length only, not generation length.
MAX_SEQ_LENGTH = 4096

# Select the best top-k value on validation set.
K_CANDIDATES = [1, 2, 3, 4]
TUNE_METRIC = "micro_f1"  # "micro_f1" or "samples_f1"

# Output directory for generated predictions and evaluation reports.
SAVE_DIR = Path("runs_pairta")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Model Configuration
# =========================

if RUN == "qwen":
    MODEL_DIR = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    MERGE_SYSTEM_TO_USER = True
    TRUST_REMOTE_CODE = True
elif RUN == "deepseek":
    MODEL_DIR = "/root/autodl-tmp/model_cache/deepseek-ai/deepseek-llm-7b-chat"
    MERGE_SYSTEM_TO_USER = True
    TRUST_REMOTE_CODE = True
elif RUN == "llama":
    MODEL_DIR = "/root/autodl-tmp/demo/base/model_cache/LLM-Research/Meta-Llama-3-8B-Instruct"
    MERGE_SYSTEM_TO_USER = False
    TRUST_REMOTE_CODE = False
else:
    raise ValueError(f"Unknown RUN: {RUN}")


RULES = (
    "Task: multi-label classification.\n"
    "Output ONLY labels from the label set.\n"
    "You may output multiple labels separated by ';'.\n"
    "Include a label only if it is clearly supported by the text. Do NOT guess.\n"
    "No explanation.\n"
)


# =========================
# File Utilities
# =========================

def get_val_test_files_pair(data_dir: Path, a: int, b: int) -> Tuple[Path, Path]:
    """Return validation and test file paths for one segment pair."""
    val_file = data_dir / f"val{a}-{b}.jsonl"
    test_file = data_dir / f"test{a}-{b}.jsonl"

    if not val_file.exists():
        raise FileNotFoundError(val_file)

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    return val_file, test_file


def get_adapter_dir(adapter_root: Path, a: int, b: int) -> Path:
    """Return adapter directory for one segment pair."""
    return adapter_root / f"pair{a}-{b}" / "adapter"


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL records."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


# =========================
# Pair-Body Truncation
# =========================

PAIR_SEP = "\n\n-----\n\n"
CHUNK_HEAD_RE = re.compile(r"(\[Chunk[^\]]*\]\n)", re.M)


def truncate_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return text

    token_ids = token_ids[:max_tokens]

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def apply_pair_body_truncation_to_user(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> List[Dict[str, str]]:
    """
    Apply training-aligned truncation to the user message.

    Only the two body chunk contents are truncated. The prompt prefix,
    title, abstract, chunk headers, and other instructions are preserved.
    """
    truncated_messages = [dict(message) for message in messages]

    user_idx = next(
        (
            idx
            for idx, message in enumerate(truncated_messages)
            if message.get("role") == "user"
        ),
        None,
    )

    if user_idx is None:
        return truncated_messages

    user_content = truncated_messages[user_idx].get("content", "")

    if PAIR_SEP not in user_content:
        return truncated_messages

    left, right = user_content.split(PAIR_SEP, 1)

    match_left = CHUNK_HEAD_RE.search(left)
    match_right = CHUNK_HEAD_RE.search(right)

    if not match_left or not match_right:
        return truncated_messages

    prefix_left = left[:match_left.start()]
    head_a = match_left.group(1)
    body_a = left[match_left.end():].strip()

    head_b = match_right.group(1)
    body_b = right[match_right.end():].strip()

    base_messages = [dict(message) for message in truncated_messages]
    dummy_user = (prefix_left + head_a).rstrip() + PAIR_SEP + head_b.rstrip()
    base_messages[user_idx]["content"] = dummy_user

    base_text = tokenizer.apply_chat_template(
        base_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    base_len = len(tokenizer.encode(base_text, add_special_tokens=False))

    buffer_tokens = 32
    budget = max(0, max_length - base_len - buffer_tokens)
    each_budget = budget // 2

    body_a_cut = truncate_tokens(tokenizer, body_a, each_budget)
    body_b_cut = truncate_tokens(tokenizer, body_b, each_budget)

    new_user = (
        (prefix_left + head_a + body_a_cut).rstrip()
        + PAIR_SEP
        + (head_b + body_b_cut).strip()
    )

    truncated_messages[user_idx]["content"] = new_user

    return truncated_messages


# =========================
# Prompt Construction
# =========================

def build_prompt(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    merge_system_to_user: bool,
    rules: str,
    max_seq_length: int,
) -> str:
    """
    Build a generation prompt from chat-style messages.

    The assistant answer is removed. Additional output rules are added.
    For Qwen and DeepSeek, the system message is merged into the user message.
    """
    messages = [message for message in messages if message["role"] != "assistant"]
    messages = [dict(message) for message in messages]

    if merge_system_to_user:
        system_text = "\n".join(
            message["content"]
            for message in messages
            if message["role"] == "system"
        ).strip()

        rest = [message for message in messages if message["role"] != "system"]

        if rest and rest[0]["role"] == "user":
            prefix = ""

            if system_text:
                prefix += system_text.strip() + "\n\n"

            prefix += rules.strip() + "\n\n"
            rest[0]["content"] = prefix + rest[0]["content"]

        messages = rest

    else:
        system_idx = next(
            (
                idx
                for idx, message in enumerate(messages)
                if message["role"] == "system"
            ),
            None,
        )

        if system_idx is None:
            messages.insert(0, {"role": "system", "content": rules.strip()})
        else:
            messages[system_idx]["content"] = (
                messages[system_idx]["content"].rstrip()
                + "\n\n"
                + rules.strip()
            ).strip()

    messages = apply_pair_body_truncation_to_user(
        messages,
        tokenizer,
        max_length=max_seq_length,
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =========================
# Generation
# =========================

def get_eos_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    """Return possible EOS token IDs for different chat templates."""
    eos_ids = []

    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    for token in ["<|eot_id|>", "<|end_of_text|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)

        if isinstance(token_id, int) and token_id != -1:
            eos_ids.append(token_id)

    return list(dict.fromkeys(eos_ids))


def run_generation(
    split_name: str,
    data_file: Path,
    out_path: Path,
    model,
    tokenizer: AutoTokenizer,
    eos_token_id,
) -> None:
    """Generate predictions for one split and save them as JSONL records."""
    dataset = load_dataset("json", data_files={split_name: str(data_file)})[split_name]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    print(f"\nGenerating {split_name} -> {out_path} (n={len(dataset)})")

    for idx, example in enumerate(tqdm(dataset)):
        prompt = build_prompt(
            example["messages"],
            tokenizer,
            MERGE_SYSTEM_TO_USER,
            RULES,
            MAX_SEQ_LENGTH,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        gen_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        gold_text = next(
            message["content"]
            for message in example["messages"]
            if message["role"] == "assistant"
        )

        record = {
            "i": idx,
            "gen_text": gen_text,
            "gold_text": gold_text,
        }

        with out_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_gen_for_one_pair(run: str, a: int, b: int) -> None:
    """Run generation for one segment pair."""
    print(f"\nGeneration for pair={a}-{b}")

    val_file, test_file = get_val_test_files_pair(DATA_DIR, a, b)
    adapter_dir = get_adapter_dir(ADAPTER_ROOT, a, b)

    if not adapter_dir.exists():
        print(f"Adapter not found. Skipping pair={a}-{b}: {adapter_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=TRUST_REMOTE_CODE,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        trust_remote_code=TRUST_REMOTE_CODE,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    eos_token_id = get_eos_token_ids(tokenizer)

    tag = f"{run}_pair{a}-{b}"
    val_out = SAVE_DIR / f"{tag}__val_gen.jsonl"
    test_out = SAVE_DIR / f"{tag}__test_gen.jsonl"

    run_generation("validation", val_file, val_out, model, tokenizer, eos_token_id)
    run_generation("test", test_file, test_out, model, tokenizer, eos_token_id)

    del model, base_model, tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    print(f"Generation finished for pair={a}-{b}")


# =========================
# Evaluation
# =========================

def eval_from_records(
    records: List[dict],
    top_k: Optional[int],
) -> Tuple[List[dict], dict, dict]:
    """Evaluate generated records against gold labels."""
    preds = []
    golds = []
    wrong_cases = []

    for record in records:
        pred_set, gold_set = pred_gold_sets(
            record["gen_text"],
            record["gold_text"],
            top_k=top_k,
        )

        preds.append(pred_set)
        golds.append(gold_set)

        if pred_set != gold_set and len(wrong_cases) < 12:
            wrong_cases.append(
                {
                    "i": record.get("i", -1),
                    "pred": sorted(pred_set),
                    "gold": sorted(gold_set),
                    "gen": record["gen_text"][:200].replace("\n", "\\n"),
                }
            )

    stats = {
        "avg_pred_labels": sum(len(item) for item in preds) / len(preds),
        "avg_gold_labels": sum(len(item) for item in golds) / len(golds),
        "empty_pred": sum(1 for item in preds if len(item) == 0),
        "n": len(preds),
        "pred_len_dist": dict(sorted(Counter(len(item) for item in preds).items())),
        "gold_len_dist": dict(sorted(Counter(len(item) for item in golds).items())),
    }

    rows, summary = compute_set_metrics_with_report(preds, golds, LABELS)

    return rows, summary, {
        "stats": stats,
        "wrong_cases": wrong_cases,
    }


def choose_best_k_on_val(val_records: List[dict]) -> int:
    """Choose the best top-k value on the validation set."""
    best_k = K_CANDIDATES[0]
    best_score = -1.0

    for k in K_CANDIDATES:
        _, summary, _ = eval_from_records(val_records, top_k=k)

        micro_f1 = summary["micro"][2]
        samples_f1 = summary["samples"][2]
        score = micro_f1 if TUNE_METRIC == "micro_f1" else samples_f1

        print(
            f"[Validation tuning] K={k} "
            f"micro_f1={micro_f1:.4f} "
            f"samples_f1={samples_f1:.4f} "
            f"score({TUNE_METRIC})={score:.4f}"
        )

        if score > best_score + 1e-12:
            best_score = score
            best_k = k

    print(f"\nBest K on validation by {TUNE_METRIC}: K={best_k} (score={best_score:.4f})")

    return best_k


def run_eval_for_one_pair(run: str, a: int, b: int) -> None:
    """Evaluate generated predictions for one segment pair."""
    tag = f"{run}_pair{a}-{b}"
    eval_txt = SAVE_DIR / f"{tag}__eval.txt"

    val_out = SAVE_DIR / f"{tag}__val_gen.jsonl"
    test_out = SAVE_DIR / f"{tag}__test_gen.jsonl"

    if not (val_out.exists() and test_out.exists()):
        raise FileNotFoundError(f"Missing generated files:\n{val_out}\n{test_out}")

    with eval_txt.open("w", encoding="utf-8") as file, redirect_stdout(file):
        val_records = load_jsonl(val_out)
        test_records = load_jsonl(test_out)

        print(f"\nLoaded records: validation={len(val_records)} test={len(test_records)}")

        best_k = choose_best_k_on_val(val_records)

        rows, summary, extra = eval_from_records(test_records, top_k=None)
        print_classification_report(rows, summary, title="TEST - RAW")

        print("\n[extra] stats:", json.dumps(extra["stats"], ensure_ascii=False))
        print(
            "\n[extra] wrong_cases:",
            json.dumps(extra["wrong_cases"], ensure_ascii=False, indent=2),
        )

        rows_tuned, summary_tuned, extra_tuned = eval_from_records(
            test_records,
            top_k=best_k,
        )
        print_classification_report(
            rows_tuned,
            summary_tuned,
            title=f"TEST - TUNED@K (K={best_k}, selected on validation)",
        )

        print("\n[extra] stats:", json.dumps(extra_tuned["stats"], ensure_ascii=False))
        print(
            "\n[extra] wrong_cases:",
            json.dumps(extra_tuned["wrong_cases"], ensure_ascii=False, indent=2),
        )

    print(f"Evaluation results saved to: {eval_txt}")


# =========================
# Pair Iteration
# =========================

def parse_pair(pair: str) -> Tuple[int, int]:
    """Parse a pair string such as '0-1' into integer indices."""
    a, b = pair.split("-")
    return int(a), int(b)


def iter_pairs_in_range(
    start_pair: str,
    end_pair: str,
    n_parts: int = 10,
):
    """Iterate over segment pairs in dictionary order."""
    start = parse_pair(start_pair)
    end = parse_pair(end_pair)

    pairs = [(i, j) for i in range(n_parts) for j in range(i + 1, n_parts)]

    started = False

    for a, b in pairs:
        current = (a, b)

        if not started:
            if current == start:
                started = True
            else:
                continue

        yield a, b

        if current == end:
            break


# =========================
# Main
# =========================

def main() -> None:
    print(f"RUN: {RUN}")
    print(f"MODE: {MODE}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ADAPTER_ROOT: {ADAPTER_ROOT}")
    print(f"PAIR RANGE: {START_PAIR} -> {END_PAIR}")
    print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"TUNE_METRIC: {TUNE_METRIC}")

    for a, b in iter_pairs_in_range(START_PAIR, END_PAIR, n_parts=10):
        print(f"\n================ Pair {a}-{b} ================")

        if MODE == "gen":
            run_gen_for_one_pair(RUN, a, b)
        elif MODE == "eval":
            run_eval_for_one_pair(RUN, a, b)
        else:
            raise ValueError("MODE must be either 'gen' or 'eval'.")


if __name__ == "__main__":
    main()