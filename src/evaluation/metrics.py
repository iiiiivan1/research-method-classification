# -*- coding: utf-8 -*-
"""
metrics.py
----------

Shared metric, decoding, and label-normalization utilities for
multi-label research method classification.

This file is used by:
- evaluate_bert.py
- evaluate_scibertlong.py
- evaluate_llm.py
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


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

LABEL_SET_LOWER = {label.lower(): label for label in LABELS}


# =========================
# Encoder Metrics and Decoding
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


# =========================
# Report Utilities for Multi-hot Labels
# =========================

def per_label_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = LABELS,
):
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

    summary = {
        "micro": (float(micro_p), float(micro_r), float(micro_f1)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_support": int(total_support),
    }

    return rows, summary


def format_report(rows, summary, title: str) -> str:
    """Format a per-label evaluation report as plain text."""
    lines = [f"\n===== {title} ====="]
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


def y_to_texts(y_bin: np.ndarray, labels: List[str] = LABELS) -> List[str]:
    """Convert binary prediction vectors to semicolon-separated label strings."""
    outputs = []

    for row_idx in range(y_bin.shape[0]):
        indices = np.where(y_bin[row_idx] == 1)[0].tolist()
        outputs.append("; ".join([labels[index] for index in indices]) if indices else "")

    return outputs


def row01_to_labels(row01: np.ndarray, labels: List[str] = LABELS) -> List[str]:
    """Convert a multi-hot vector to label names."""
    indices = np.where(row01.astype(int) == 1)[0].tolist()
    return [labels[index] for index in indices]


def labels_to_text(labels: List[str]) -> str:
    """Convert a list of labels to a semicolon-separated string."""
    return "; ".join(labels) if labels else ""


# =========================
# LLM Label Parsing
# =========================

def extract_labels_ordered(text: str, max_lines: Optional[int] = None) -> List[str]:
    """
    Extract method labels from generated text.

    This function preserves the output order and normalizes label strings
    to the predefined label set.
    """
    if not text:
        return []

    raw_text = text.strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if max_lines is not None:
        lines = lines[:max_lines]

    joined = "\n".join(lines)
    joined = re.sub(
        r"^(labels?\s*[:\-]\s*)",
        "",
        joined.strip(),
        flags=re.IGNORECASE,
    )
    joined = re.sub(r"[,\n，；|/]+", ";", joined)

    ordered_labels = []
    seen = set()

    for token in [token.strip() for token in joined.split(";") if token.strip()]:
        key = token.lower()

        if key in LABEL_SET_LOWER:
            label = LABEL_SET_LOWER[key]

            if label not in seen:
                ordered_labels.append(label)
                seen.add(label)

    if not ordered_labels:
        for label in LABELS:
            pattern = r"(?<![A-Za-z])" + re.escape(label) + r"(?![A-Za-z])"

            if re.search(pattern, joined, flags=re.IGNORECASE):
                if label not in seen:
                    ordered_labels.append(label)
                    seen.add(label)

    return ordered_labels


def pred_gold_sets(
    gen_text: str,
    gold_text: str,
    top_k: Optional[int] = None,
) -> Tuple[Set[str], Set[str]]:
    """Convert generated and gold label strings into prediction and gold sets."""
    pred_list = extract_labels_ordered(gen_text, max_lines=2)
    gold_list = extract_labels_ordered(gold_text, max_lines=None)

    if top_k is not None:
        pred_list = pred_list[:top_k]

    return set(pred_list), set(gold_list)


def compute_set_metrics_with_report(
    preds: List[Set[str]],
    golds: List[Set[str]],
    labels: List[str] = LABELS,
):
    """
    Compute micro, macro, weighted, and sample-averaged metrics
    for set-form predictions.

    This is mainly used for LLM-generated label strings.
    """
    per_label = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in labels
    }

    for pred_set, gold_set in zip(preds, golds):
        for label in labels:
            in_pred = label in pred_set
            in_gold = label in gold_set

            if in_gold:
                per_label[label]["support"] += 1

            if in_pred and in_gold:
                per_label[label]["tp"] += 1
            elif in_pred and not in_gold:
                per_label[label]["fp"] += 1
            elif not in_pred and in_gold:
                per_label[label]["fn"] += 1

    rows = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    total_support = 0

    for label in labels:
        stats = per_label[label]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        support = stats["support"]

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append(
            {
                "label": label,
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

    macro_p = sum(row["precision"] for row in rows) / len(rows)
    macro_r = sum(row["recall"] for row in rows) / len(rows)
    macro_f1 = sum(row["f1"] for row in rows) / len(rows)

    weighted_f1 = (
        sum(row["f1"] * row["support"] for row in rows)
        / (total_support + 1e-9)
    )

    sample_precisions = []
    sample_recalls = []
    sample_f1s = []

    for pred_set, gold_set in zip(preds, golds):
        if not pred_set and not gold_set:
            sample_precisions.append(1.0)
            sample_recalls.append(1.0)
            sample_f1s.append(1.0)
            continue

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        sample_p = tp / (tp + fp + 1e-9)
        sample_r = tp / (tp + fn + 1e-9)
        sample_f1 = 2 * sample_p * sample_r / (sample_p + sample_r + 1e-9)

        sample_precisions.append(sample_p)
        sample_recalls.append(sample_r)
        sample_f1s.append(sample_f1)

    samples_p = sum(sample_precisions) / len(sample_precisions)
    samples_r = sum(sample_recalls) / len(sample_recalls)
    samples_f1 = sum(sample_f1s) / len(sample_f1s)

    summary = {
        "micro": (micro_p, micro_r, micro_f1),
        "macro": (macro_p, macro_r, macro_f1),
        "weighted_f1": weighted_f1,
        "samples": (samples_p, samples_r, samples_f1),
        "total_support": total_support,
    }

    return rows, summary


def print_classification_report(rows, summary, title: str = "") -> None:
    """Print a text classification report."""
    if title:
        print(f"\n===== {title} =====\n")

    print(
        f"{'Label':30s} {'Precision':>10s} "
        f"{'Recall':>8s} {'F1-score':>10s} {'Support':>9s}"
    )
    print("-" * 75)

    for row in rows:
        print(
            f"{row['label']:30s} "
            f"{row['precision']:10.4f} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:10.4f} "
            f"{row['support']:9d}"
        )

    print("-" * 75)

    micro_p, micro_r, micro_f1 = summary["micro"]
    total = summary["total_support"]

    print(
        f"{'micro avg':30s} "
        f"{micro_p:10.4f} {micro_r:8.4f} {micro_f1:10.4f} {total:9d}"
    )

    if "macro" in summary:
        macro_p, macro_r, macro_f1 = summary["macro"]
        print(
            f"{'macro avg':30s} "
            f"{macro_p:10.4f} {macro_r:8.4f} {macro_f1:10.4f} {total:9d}"
        )
    elif "macro_f1" in summary:
        print(
            f"{'macro_f1':30s} "
            f"{'':10s} {'':8s} {summary['macro_f1']:10.4f} {total:9d}"
        )

    if "weighted_f1" in summary:
        print(
            f"{'weighted avg':30s} "
            f"{'':10s} {'':8s} {summary['weighted_f1']:10.4f} {total:9d}"
        )

    if "samples" in summary:
        samples_p, samples_r, samples_f1 = summary["samples"]
        print(
            f"{'samples avg':30s} "
            f"{samples_p:10.4f} {samples_r:8.4f} {samples_f1:10.4f} {total:9d}"
        )