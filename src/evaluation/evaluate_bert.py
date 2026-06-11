# -*- coding: utf-8 -*-
"""
evaluate_bert.py
----------------

Evaluation script for BERT-based P4 experiments.

This script evaluates a trained LoRA adapter on P4 inputs:
Title + Abstract + Segment_i + Segment_j.

Main functions:
- Load a trained LoRA adapter.
- Load train / validation / test JSONL files.
- Apply P4 fair tokenization.
- Search decoding strategies on the validation set.
- Evaluate the selected strategy on the test set.
- Export logits, probabilities, labels, and text-form predictions.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Enable online access by default.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from metrics import (
    LABELS,
    sigmoid_np,
    micro_scores,
    pred_threshold,
    pred_topk,
    apply_rule,
    per_label_report,
    format_report,
    row01_to_labels,
    labels_to_text,
)


# =========================
# Configuration
# =========================

MODEL_NAME = "bert-base-uncased"
EXPERIMENT_MODE = "P4"

# Normalize PAIR_ID.
# Example:
#   PAIR_ID=6-9 python evaluate_bert.py
# will be normalized to 06-09.
raw_pair = os.environ.get("PAIR_ID", "00-01")
left_id, right_id = raw_pair.split("-")
PAIR_ID = f"{int(left_id):02d}-{int(right_id):02d}"

# Project root:
# This file is expected to be placed at src/evaluation/evaluate_bert.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data and model paths.
DATA_DIR = (PROJECT_ROOT / "out_bert_p4_pairs_ta").resolve()
TRAIN_FILE = DATA_DIR / f"train_{PAIR_ID}.jsonl"
VAL_FILE = DATA_DIR / f"val_{PAIR_ID}.jsonl"
TEST_FILE = DATA_DIR / f"test_{PAIR_ID}.jsonl"

ADAPTER_DIR = (
    PROJECT_ROOT / f"outputs/bert/{EXPERIMENT_MODE.lower()}_{PAIR_ID}/adapter"
).resolve()

SAVE_DIR = (PROJECT_ROOT / f"eval_runs/bert_p4_{PAIR_ID}").resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = SAVE_DIR / "eval_report.txt"

MAX_LEN = 512
TA_MAX_TOKENS = int(os.environ.get("TA_MAX_TOKENS", "160"))
FAIR_SPLIT = True
EVAL_BS = int(os.environ.get("EVAL_BS", "64"))

THRS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]
K_LIST = [1, 2, 3, 4, 5]
MAXK_LIST = [3, 4, 5, 6]

NUM_LABELS = len(LABELS)


# =========================
# P4 Tokenization
# =========================

def ensure_list(value, n: int) -> List[str]:
    """Convert a possibly missing field into a list of strings."""
    if value is None:
        return [""] * n

    return [("" if item is None else str(item)) for item in value]


def tokenize_p4_fair(examples, tokenizer):
    """
    Tokenize P4 input with fair token allocation.

    Input structure:
        [CLS] Title + Abstract [SEP] Segment_i [SEP] Segment_j [SEP]

    The title and abstract are kept within TA_MAX_TOKENS.
    The remaining token budget is evenly allocated to the two body segments.
    """
    chunk_i_list = (
        examples.get("chunk_i")
        or examples.get("seg_i")
        or examples.get("text_i")
    )
    chunk_j_list = (
        examples.get("chunk_j")
        or examples.get("seg_j")
        or examples.get("text_j")
    )

    n = len(chunk_i_list) if chunk_i_list else 1

    titles = ensure_list(examples.get("title"), n)
    abstracts = ensure_list(examples.get("abstract"), n)
    chunk_i_list = ensure_list(chunk_i_list, n)
    chunk_j_list = ensure_list(chunk_j_list, n)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # [CLS] TA [SEP] Segment_i [SEP] Segment_j [SEP]
    num_special_tokens = 4

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []

    for title, abstract, chunk_i, chunk_j in zip(
        titles,
        abstracts,
        chunk_i_list,
        chunk_j_list,
    ):
        title = title.strip()
        abstract = abstract.strip()
        chunk_i = chunk_i.strip()
        chunk_j = chunk_j.strip()

        if FAIR_SPLIT:
            ta_text = (title + "\n" + abstract).strip()

            encoded_ta = tokenizer(
                ta_text,
                truncation=True,
                max_length=TA_MAX_TOKENS,
                padding=False,
                add_special_tokens=False,
            )
            ta_ids = encoded_ta["input_ids"]

            remaining_budget = MAX_LEN - (len(ta_ids) + num_special_tokens)

            if remaining_budget < 2:
                keep_len = max(0, len(ta_ids) - (2 - remaining_budget))
                ta_ids = ta_ids[:keep_len]
                remaining_budget = 2

            len_i = remaining_budget // 2
            len_j = remaining_budget - len_i

            encoded_i = tokenizer(
                chunk_i,
                truncation=True,
                max_length=len_i,
                padding=False,
                add_special_tokens=False,
            )
            encoded_j = tokenizer(
                chunk_j,
                truncation=True,
                max_length=len_j,
                padding=False,
                add_special_tokens=False,
            )

            input_ids = (
                [cls_id]
                + ta_ids
                + [sep_id]
                + encoded_i["input_ids"]
                + [sep_id]
                + encoded_j["input_ids"]
                + [sep_id]
            )

            attention_mask = [1] * len(input_ids)

            # BERT token type IDs:
            # 0 for Title + Abstract, 1 for body segments.
            token_type_ids = (
                [0] * (1 + len(ta_ids) + 1)
                + [1]
                * (
                    len(encoded_i["input_ids"])
                    + 1
                    + len(encoded_j["input_ids"])
                    + 1
                )
            )

        else:
            text = (
                title
                + "\n"
                + abstract
                + "\n\n"
                + chunk_i
                + "\n\n"
                + chunk_j
            ).strip()

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))

        input_ids_list.append(input_ids[:MAX_LEN])
        attention_mask_list.append(attention_mask[:MAX_LEN])
        token_type_ids_list.append(token_type_ids[:MAX_LEN])

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "token_type_ids": token_type_ids_list,
    }


# =========================
# Export Utilities
# =========================

def export_gen_jsonl(
    path: Path,
    y_pred01: np.ndarray,
    y_true01: np.ndarray,
) -> None:
    """Export prediction and gold labels as JSONL text records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        n_rows = int(y_pred01.shape[0])

        for row_idx in range(n_rows):
            pred_labels = row01_to_labels(y_pred01[row_idx], LABELS)
            gold_labels = row01_to_labels(y_true01[row_idx], LABELS)

            row = {
                "i": int(row_idx),
                "gen_text": labels_to_text(pred_labels),
                "gold_text": labels_to_text(gold_labels),
            }

            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_split_dir(
    split_name: str,
    y_true01: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
    pred_best01: np.ndarray,
    pred_raw_thr01: np.ndarray,
    pred_topk01: np.ndarray,
) -> None:
    """Save arrays and text-form predictions for one dataset split."""
    output_dir = SAVE_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "labels.npy", y_true01)
    np.save(output_dir / "probs.npy", probs)
    np.save(output_dir / "logits.npy", logits)
    np.save(output_dir / "pred_best.npy", pred_best01)
    np.save(output_dir / "pred_raw_thr.npy", pred_raw_thr01)
    np.save(output_dir / "pred_topk.npy", pred_topk01)

    export_gen_jsonl(output_dir / "pred_best_gen.jsonl", pred_best01, y_true01)
    export_gen_jsonl(output_dir / "pred_raw_thr_gen.jsonl", pred_raw_thr01, y_true01)
    export_gen_jsonl(output_dir / "pred_topk_gen.jsonl", pred_topk01, y_true01)


# =========================
# Main
# =========================

def main() -> None:
    print(f"CWD: {Path.cwd()}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"PAIR_ID: {PAIR_ID}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ADAPTER_DIR: {ADAPTER_DIR}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print(f"EVAL_BS: {EVAL_BS}")

    assert VAL_FILE.exists(), f"Validation file not found: {VAL_FILE}"
    assert ADAPTER_DIR.exists(), f"Adapter directory not found: {ADAPTER_DIR}"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    print("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
    ).merge_and_unload().to(device)

    model.eval()

    data_files = {"val": str(VAL_FILE)}

    if TEST_FILE.exists():
        data_files["test"] = str(TEST_FILE)

    if TRAIN_FILE.exists():
        data_files["train"] = str(TRAIN_FILE)

    print(f"Loading dataset: {data_files}")

    dataset = load_dataset("json", data_files=data_files)

    remove_columns = [
        column for column in dataset["val"].column_names
        if column != "labels"
    ]

    dataset = dataset.map(
        lambda batch: tokenize_p4_fair(batch, tokenizer),
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing P4 inputs",
    )

    def cast_labels(batch):
        batch["labels"] = [[float(value) for value in y] for y in batch["labels"]]
        return batch

    dataset = dataset.map(
        cast_labels,
        batched=True,
        desc="Casting labels",
    )

    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features):
        labels = [feature.pop("labels") for feature in features]
        batch = base_collator(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "tmp_eval"),
        per_device_eval_batch_size=EVAL_BS if device.type == "cuda" else 32,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device.type == "cuda"),
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    outputs = {}

    for split in dataset.keys():
        print(f"Predicting logits for {split} split (n={len(dataset[split])})...")
        outputs[split] = trainer.predict(dataset[split])

    y_val = (outputs["val"].label_ids > 0.5).astype(int)
    p_val = sigmoid_np(outputs["val"].predictions)

    print("\nGrid search on the validation set using Micro-F1...")

    best_overall = None
    all_candidates: List[Dict[str, Any]] = []

    def update_best(best, candidate):
        return (
            candidate
            if best is None or candidate["micro_f1"] > best["micro_f1"]
            else best
        )

    for threshold in THRS:
        candidate = {
            "rule": "thr",
            "thr": float(threshold),
            **micro_scores(y_val, pred_threshold(p_val, threshold)),
        }
        all_candidates.append(candidate)
        best_overall = update_best(best_overall, candidate)

    for k in K_LIST:
        candidate = {
            "rule": "topk",
            "k": int(k),
            **micro_scores(y_val, pred_topk(p_val, k)),
        }
        all_candidates.append(candidate)
        best_overall = update_best(best_overall, candidate)

    for threshold in THRS:
        for max_k in MAXK_LIST:
            candidate = {
                "rule": "hybrid",
                "thr": float(threshold),
                "max_k": int(max_k),
                **micro_scores(y_val, pred_hybrid(p_val, threshold, max_k)),
            }
            all_candidates.append(candidate)
            best_overall = update_best(best_overall, candidate)

    print(f"\nBest validation strategy: {json.dumps(best_overall, ensure_ascii=False)}")

    with (SAVE_DIR / "best_strategy.json").open("w", encoding="utf-8") as file:
        json.dump(best_overall, file, ensure_ascii=False, indent=2)

    if "test" in outputs:
        y_test = (outputs["test"].label_ids > 0.5).astype(int)
        p_test = sigmoid_np(outputs["test"].predictions)
        y_pred_test = apply_rule(p_test, best_overall)

        rows, summary = per_label_report(y_test, y_pred_test, LABELS)
        report = format_report(
            rows,
            summary,
            f"TEST REPORT (BERT P4, PAIR_ID={PAIR_ID}, Rule={best_overall['rule']})",
        )

        print(report)

        with REPORT_TXT.open("w", encoding="utf-8") as file:
            file.write(report + "\n")

    else:
        with REPORT_TXT.open("w", encoding="utf-8") as file:
            file.write(
                f"TEST_FILE missing; only validation tuning was performed. "
                f"PAIR_ID={PAIR_ID}\n"
            )

    threshold_for_export = (
        float(best_overall["thr"])
        if best_overall["rule"] in ["thr", "hybrid"]
        else 0.5
    )
    k_for_export = int(best_overall["k"]) if best_overall["rule"] == "topk" else 1

    for split in outputs.keys():
        y_true = (outputs[split].label_ids > 0.5).astype(int)
        logits = outputs[split].predictions
        probs = sigmoid_np(logits)

        pred_best = apply_rule(probs, best_overall)
        pred_raw = pred_threshold(probs, threshold_for_export)
        pred_topk_arr = pred_topk(probs, k_for_export)

        save_split_dir(
            split_name=split,
            y_true01=y_true,
            probs=probs,
            logits=logits,
            pred_best01=pred_best,
            pred_raw_thr01=pred_raw,
            pred_topk01=pred_topk_arr,
        )

    print(f"\nEvaluation finished. Results saved to: {SAVE_DIR}")
    print(
        "Generated files include best_strategy.json, eval_report.txt, "
        "and split-level prediction outputs."
    )


if __name__ == "__main__":
    main()