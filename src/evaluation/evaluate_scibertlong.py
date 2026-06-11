# -*- coding: utf-8 -*-
"""
evaluate_scibertlong.py
-----------------------

Evaluation script for SciBERT-long P4 experiments.

This script evaluates trained LoRA adapters on P4 inputs:
Title + Abstract + Segment_i + Segment_j.

Main functions:
- Load validation and test JSONL files.
- Load and merge trained LoRA adapters.
- Apply SciBERT-long tokenization with global attention masks.
- Tune decoding strategies on the validation set.
- Evaluate the selected decoding strategy on the test set.
- Export prediction texts, logits, and evaluation reports.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# =========================
# Configuration
# =========================

MODEL_NAME = "yorko/scibert_scivocab_uncased_long_4096"

# Directory containing P4 pair data:
# train_00-01.jsonl / val_00-01.jsonl / test_00-01.jsonl
DATA_DIR = Path("./out_pairs_p4_scilong_titleabs")

# Directory containing trained adapters:
# outputs_p4_pairs/P4_00-01/adapter
OUT_ROOT = Path("./outputs_p4_pairs")

# Evaluation output directory.
EVAL_ROOT = Path("./eval_runs_p4_pairs")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

MAX_LEN = 4096

# Candidate decoding strategies.
THRS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]
K_LIST = [1, 2, 3, 4, 5]
MAXK_LIST = [3, 4, 5, 6]

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

NUM_LABELS = len(LABELS)

# Optional: evaluate only one pair.
# Example:
#   PAIR_ID=00-01 python evaluate_scibertlong.py
ONLY_PAIR = os.environ.get("PAIR_ID", "").strip()

# Optional: evaluate a range of pairs.
# Example:
#   START_PAIR=00-01 END_PAIR=03-04 python evaluate_scibertlong.py
START_PAIR = os.environ.get("START_PAIR", "").strip()
END_PAIR = os.environ.get("END_PAIR", "").strip()

PER_DEVICE_EVAL_BS = int(os.environ.get("EVAL_BS", "2"))

# Truncation mode:
# - "longest": use tokenizer longest-first truncation.
# - "budget": allocate separate token budgets to title+abstract, chunk_i, and chunk_j.
TRUNC_MODE = os.environ.get("TRUNC_MODE", "budget").strip().lower()

# Optional token budgets. If set to 0, default budgets are used.
MAX_TA = int(os.environ.get("MAX_TA", "0"))
MAX_I = int(os.environ.get("MAX_I", "0"))
MAX_J = int(os.environ.get("MAX_J", "0"))


# =========================
# Metrics and Decoding Utilities
# =========================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function to a NumPy array."""
    return 1.0 / (1.0 + np.exp(-x))


def micro_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute micro-averaged precision, recall, and F1."""
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_p": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_r": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
    }


def pred_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities into binary predictions using a fixed threshold."""
    return (probs >= threshold).astype(int)


def pred_topk(probs: np.ndarray, k: int) -> np.ndarray:
    """Predict the top-k labels for each sample."""
    top_indices = np.argsort(-probs, axis=1)[:, :k]
    y_pred = np.zeros_like(probs, dtype=int)

    for row_idx in range(y_pred.shape[0]):
        y_pred[row_idx, top_indices[row_idx]] = 1

    return y_pred


def pred_hybrid(probs: np.ndarray, threshold: float, max_k: int) -> np.ndarray:
    """
    Hybrid decoding strategy.

    Labels above the threshold are selected.
    If no label is selected, the top-1 label is used.
    If too many labels are selected, only the top max_k labels are kept.
    """
    y_pred = (probs >= threshold).astype(int)
    sorted_indices = np.argsort(-probs, axis=1)

    for row_idx in range(y_pred.shape[0]):
        selected_count = int(y_pred[row_idx].sum())

        if selected_count == 0:
            y_pred[row_idx, sorted_indices[row_idx, 0]] = 1
            selected_count = 1

        if selected_count > max_k:
            y_pred[row_idx, :] = 0
            y_pred[row_idx, sorted_indices[row_idx, :max_k]] = 1

    return y_pred


def apply_rule(probs: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a decoding rule to probability outputs."""
    if rule["rule"] == "thr":
        return pred_threshold(probs, float(rule["thr"]))

    if rule["rule"] == "topk":
        return pred_topk(probs, int(rule["k"]))

    return pred_hybrid(probs, float(rule["thr"]), int(rule["max_k"]))


def per_label_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    """Compute per-label precision, recall, F1, and support."""
    rows = []
    total_support = 0
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for label_idx, label_name in enumerate(labels):
        gold = y_true[:, label_idx].astype(int)
        pred = y_pred[:, label_idx].astype(int)

        tp = int(np.sum((gold == 1) & (pred == 1)))
        fp = int(np.sum((gold == 0) & (pred == 1)))
        fn = int(np.sum((gold == 1) & (pred == 0)))
        support = int(np.sum(gold == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append(
            {
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        total_support += support

    micro_p = micro_tp / (micro_tp + micro_fp + 1e-9)
    micro_r = micro_tp / (micro_tp + micro_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    macro_f1 = float(np.mean([row["f1"] for row in rows]))
    weighted_f1 = float(
        np.sum([row["f1"] * row["support"] for row in rows])
        / (total_support + 1e-9)
    )

    return rows, {
        "micro": (float(micro_p), float(micro_r), float(micro_f1)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_support": int(total_support),
    }


def format_report(rows, summary, title: str) -> str:
    """Format a per-label evaluation report as plain text."""
    lines = [f"\n===== {title} =====\n"]
    lines.append(
        f"{'Label':30s} {'Precision':>10s} "
        f"{'Recall':>8s} {'F1-score':>10s} {'Support':>9s}"
    )
    lines.append("-" * 75)

    for row in rows:
        lines.append(
            f"{row['label']:30s} "
            f"{row['precision']:10.4f} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:10.4f} "
            f"{row['support']:9d}"
        )

    lines.append("-" * 75)

    micro_p, micro_r, micro_f1 = summary["micro"]
    total = summary["total_support"]

    lines.append(
        f"{'micro avg':30s} {micro_p:10.4f} "
        f"{micro_r:8.4f} {micro_f1:10.4f} {total:9d}"
    )
    lines.append(
        f"{'macro_f1':30s} {'':10s} "
        f"{'':8s} {summary['macro_f1']:10.4f} {total:9d}"
    )
    lines.append(
        f"{'weighted_f1':30s} {'':10s} "
        f"{'':8s} {summary['weighted_f1']:10.4f} {total:9d}"
    )

    return "\n".join(lines)


def y_to_texts(y_bin: np.ndarray, labels: List[str]) -> List[str]:
    """Convert binary prediction vectors to semicolon-separated label strings."""
    outputs = []

    for row_idx in range(y_bin.shape[0]):
        indices = np.where(y_bin[row_idx] == 1)[0].tolist()
        outputs.append("; ".join([labels[index] for index in indices]) if indices else "")

    return outputs


def save_gen_gold_jsonl(
    save_path: Path,
    gen_texts: List[str],
    gold_texts: List[str],
) -> None:
    """Save predicted labels and gold labels as JSONL text records."""
    assert len(gen_texts) == len(gold_texts)

    with save_path.open("w", encoding="utf-8") as file:
        for row_idx, (gen_text, gold_text) in enumerate(zip(gen_texts, gold_texts)):
            record = {
                "i": row_idx,
                "gen_text": gen_text,
                "gold_text": gold_text,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# P4 Preprocessing
# =========================

def budget_defaults(max_len: int):
    """
    Return default token budgets for P4 input.

    For max_len=4096, the default allocation is approximately:
        Title + Abstract: 1024 tokens
        Segment i: 1536 tokens
        Segment j: 1536 tokens
    """
    ta_budget = min(1024, max_len // 4)
    remaining = max_len - ta_budget
    i_budget = remaining // 2
    j_budget = remaining - i_budget

    return ta_budget, i_budget, j_budget


def preprocess(examples, tokenizer):
    """
    Tokenize P4 samples for SciBERT-long.

    Input structure:
        A = Title + Abstract + Segment i
        B = Segment j

    Global attention:
        CLS token and the A segment up to the first SEP token are assigned
        global attention.
    """
    titles = examples.get("title", [""] * len(examples["chunk_i"]))
    abstracts = examples.get("abstract", [""] * len(examples["chunk_i"]))
    chunk_is = examples.get("chunk_i", [""] * len(examples["chunk_i"]))
    chunk_js = examples.get("chunk_j", [""] * len(examples["chunk_i"]))

    if isinstance(titles, str):
        titles = [titles]
    if isinstance(abstracts, str):
        abstracts = [abstracts]
    if isinstance(chunk_is, str):
        chunk_is = [chunk_is]
    if isinstance(chunk_js, str):
        chunk_js = [chunk_js]

    input_ids_list = []
    attention_mask_list = []
    global_attention_mask_list = []

    default_ta, default_i, default_j = budget_defaults(MAX_LEN)

    ta_budget = MAX_TA if MAX_TA > 0 else default_ta
    i_budget = MAX_I if MAX_I > 0 else default_i
    j_budget = MAX_J if MAX_J > 0 else default_j

    sep_id = tokenizer.sep_token_id

    for title, abstract, chunk_i, chunk_j in zip(titles, abstracts, chunk_is, chunk_js):
        title = title or ""
        abstract = abstract or ""
        chunk_i = chunk_i or ""
        chunk_j = chunk_j or ""

        if TRUNC_MODE == "longest":
            a_text = title.strip()

            if abstract.strip():
                a_text = (
                    a_text + "\n\n" + abstract.strip()
                ).strip() if a_text else abstract.strip()

            if chunk_i.strip():
                a_text = (
                    a_text + "\n\n[Body Segment i]\n" + chunk_i.strip()
                ).strip() if a_text else chunk_i.strip()

            encoded = tokenizer(
                a_text,
                chunk_j.strip(),
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )

        else:
            ta_text = title.strip()

            if abstract.strip():
                ta_text = (
                    ta_text + "\n\n" + abstract.strip()
                ).strip() if ta_text else abstract.strip()

            ta_ids = tokenizer(
                ta_text,
                truncation=True,
                max_length=ta_budget,
                padding=False,
            )["input_ids"]

            i_ids = tokenizer(
                chunk_i.strip(),
                truncation=True,
                max_length=i_budget,
                padding=False,
            )["input_ids"]

            j_ids = tokenizer(
                chunk_j.strip(),
                truncation=True,
                max_length=j_budget,
                padding=False,
            )["input_ids"]

            ta_text_cut = tokenizer.decode(ta_ids, skip_special_tokens=True).strip()
            i_text_cut = tokenizer.decode(i_ids, skip_special_tokens=True).strip()
            j_text_cut = tokenizer.decode(j_ids, skip_special_tokens=True).strip()

            a_text = ta_text_cut

            if i_text_cut:
                a_text = (
                    a_text + "\n\n[Body Segment i]\n" + i_text_cut
                ).strip() if a_text else i_text_cut

            encoded = tokenizer(
                a_text,
                j_text_cut,
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        global_attention_mask = [0] * len(input_ids)

        if global_attention_mask:
            global_attention_mask[0] = 1

        if sep_id is not None:
            try:
                sep_index = input_ids.index(sep_id)

                for token_idx in range(sep_index + 1):
                    global_attention_mask[token_idx] = 1

            except ValueError:
                pass

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        global_attention_mask_list.append(global_attention_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "global_attention_mask": global_attention_mask_list,
    }


# =========================
# Pair Discovery
# =========================

def list_pairs(data_dir: Path) -> List[str]:
    """List all available pair IDs from the data directory."""
    import re

    pattern = re.compile(r"^val_(\d{2}-\d{2})\.jsonl$")
    pairs = []

    for path in data_dir.iterdir():
        if path.is_file():
            match = pattern.match(path.name)

            if match:
                pairs.append(match.group(1))

    pairs = sorted(pairs)

    if not pairs:
        raise FileNotFoundError(f"No val_??-??.jsonl files found in {data_dir}")

    return pairs


def pick_pairs(all_pairs: List[str]) -> List[str]:
    """Select pair IDs according to environment variables."""
    if ONLY_PAIR:
        if ONLY_PAIR not in all_pairs:
            raise FileNotFoundError(
                f"PAIR_ID={ONLY_PAIR} not found. Example pairs: {all_pairs[:8]}"
            )
        return [ONLY_PAIR]

    if START_PAIR and END_PAIR:
        if START_PAIR not in all_pairs or END_PAIR not in all_pairs:
            raise FileNotFoundError(
                f"START_PAIR or END_PAIR not found. "
                f"START_PAIR={START_PAIR}, END_PAIR={END_PAIR}"
            )

        start_idx = all_pairs.index(START_PAIR)
        end_idx = all_pairs.index(END_PAIR)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        return all_pairs[start_idx:end_idx + 1]

    return all_pairs


# =========================
# Evaluation
# =========================

def evaluate_one(pair_name: str, device: torch.device) -> None:
    """Evaluate one SciBERT-long P4 segment pair."""
    val_file = DATA_DIR / f"val_{pair_name}.jsonl"
    test_file = DATA_DIR / f"test_{pair_name}.jsonl"

    output_name = f"P4_{pair_name}"
    adapter_dir = OUT_ROOT / output_name / "adapter"

    save_dir = EVAL_ROOT / output_name
    save_dir.mkdir(parents=True, exist_ok=True)

    report_txt = save_dir / "eval_report.txt"
    pred_texts_test = save_dir / "pred_texts_test.jsonl"
    pred_texts_val = save_dir / "pred_texts_val.jsonl"

    if not val_file.exists() or not test_file.exists():
        print(f"[{output_name}] Missing validation or test JSONL file. Skipping.")
        print(f"  validation file: {val_file}")
        print(f"  test file: {test_file}")
        return

    if not adapter_dir.exists():
        print(f"[{output_name}] Missing adapter directory. Skipping: {adapter_dir}")
        return

    print("\n" + "=" * 88)
    print(f"Evaluation: {output_name} | device={device}")
    print(f"Validation file: {val_file}")
    print(f"Test file: {test_file}")
    print(f"Adapter directory: {adapter_dir}")
    print(f"Save directory: {save_dir}")
    print(
        f"TRUNC_MODE={TRUNC_MODE} | MAX_LEN={MAX_LEN} | "
        f"(MAX_TA, MAX_I, MAX_J)=({MAX_TA}, {MAX_I}, {MAX_J})"
    )
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    print("Merging LoRA adapter for inference...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
    ).merge_and_unload().to(device)

    model.eval()

    dataset = load_dataset(
        "json",
        data_files={
            "val": str(val_file),
            "test": str(test_file),
        },
    )

    remove_columns = [
        column for column in dataset["val"].column_names
        if column != "labels"
    ]

    dataset = dataset.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=remove_columns,
        desc=f"Tokenizing {output_name}",
    )

    def cast_labels(batch):
        batch["labels"] = [[float(value) for value in y] for y in batch["labels"]]
        return batch

    dataset = dataset.map(
        cast_labels,
        batched=True,
        desc="Casting labels",
    )

    base_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    def collator(features):
        labels = [feature.pop("labels") for feature in features]
        batch = base_collator(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    training_args = TrainingArguments(
        output_dir=str(save_dir / "tmp_eval"),
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        dataloader_num_workers=2,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(
        f"\nPredicting logits for validation "
        f"({len(dataset['val'])}) and test ({len(dataset['test'])})..."
    )

    val_output = trainer.predict(dataset["val"])
    test_output = trainer.predict(dataset["test"])

    y_val = (val_output.label_ids > 0.5).astype(int)
    y_test = (test_output.label_ids > 0.5).astype(int)

    p_val = sigmoid_np(val_output.predictions)
    p_test = sigmoid_np(test_output.predictions)

    print("\nGrid search on validation set using Micro-F1...")

    best_overall = None
    all_candidates = []

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

    all_candidates_sorted = sorted(
        all_candidates,
        key=lambda item: item["micro_f1"],
        reverse=True,
    )

    print(f"Best validation strategy: {json.dumps(best_overall, ensure_ascii=False)}")

    print("\nTop-5 validation strategies:")
    for rank, candidate in enumerate(all_candidates_sorted[:5], 1):
        brief = {
            key: candidate[key]
            for key in ["rule", "thr", "k", "max_k", "micro_f1", "avg_pred_labels"]
            if key in candidate
        }
        print(f"[{rank}] {brief}")

    y_pred_val = apply_rule(p_val, best_overall)
    y_pred_test = apply_rule(p_test, best_overall)

    gen_val = y_to_texts(y_pred_val, LABELS)
    gold_val = y_to_texts(y_val, LABELS)

    gen_test = y_to_texts(y_pred_test, LABELS)
    gold_test = y_to_texts(y_test, LABELS)

    save_gen_gold_jsonl(pred_texts_val, gen_val, gold_val)
    save_gen_gold_jsonl(pred_texts_test, gen_test, gold_test)

    print("\nApplying best strategy to test set...")

    rows, summary = per_label_report(y_test, y_pred_test, LABELS)

    report = format_report(
        rows,
        summary,
        f"TEST REPORT ({output_name}, Rule: {best_overall['rule']})",
    )

    print(report)

    np.save(save_dir / "test_preds.npy", y_pred_test)
    np.save(save_dir / "val_logits.npy", val_output.predictions)
    np.save(save_dir / "test_logits.npy", test_output.predictions)

    with (save_dir / "best_strategy.json").open("w", encoding="utf-8") as file:
        json.dump(best_overall, file, ensure_ascii=False, indent=2)

    with report_txt.open("w", encoding="utf-8") as file:
        file.write(report + "\n")

    print(f"\nEvaluation finished. Results saved to: {save_dir}")
    print(f"Test prediction JSONL: {pred_texts_test}")
    print(f"Validation prediction JSONL: {pred_texts_val}")


# =========================
# Main
# =========================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device} | Model: {MODEL_NAME}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"EVAL_ROOT: {EVAL_ROOT}")
    print(f"MAX_LEN: {MAX_LEN} | EVAL_BS: {PER_DEVICE_EVAL_BS}")
    print(f"TRUNC_MODE: {TRUNC_MODE} | MAX_TA={MAX_TA} MAX_I={MAX_I} MAX_J={MAX_J}")

    all_pairs = list_pairs(DATA_DIR)
    run_pairs = pick_pairs(all_pairs)

    print(f"Found pairs: {len(all_pairs)}")
    print("Will run pairs:", run_pairs[:10], "..." if len(run_pairs) > 10 else "")

    for pair in run_pairs:
        evaluate_one(pair, device=device)


if __name__ == "__main__":
    main()