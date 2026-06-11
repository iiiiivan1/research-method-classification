# -*- coding: utf-8 -*-

import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import re
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


# =========================
# Runtime Settings
# =========================

RUN = "qwen"
SKIP_DONE = True

# Directory containing pair-wise instruction-tuning data.
DATA_DIR = Path(f"out_10ta_{RUN}")

# Root directory for model outputs.
OUTPUT_ROOT = Path("outputs_pairta") / RUN
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

N_PARTS = 10
PAIRS = [(i, j) for i in range(N_PARTS) for j in range(i + 1, N_PARTS)]

# Maximum sequence length for instruction tuning.
# Use 4096 for memory-limited GPUs, or 8192 if memory is sufficient.
MAX_SEQ_LENGTH = 4096


# =========================
# Model Configuration
# =========================

@dataclass(frozen=True)
class RuntimeCfg:
    """Runtime configuration for an LLM backbone."""

    run: str
    model_dir: str
    trust_remote_code: bool
    local_files_only: bool


MODEL_REGISTRY = {
    "qwen": RuntimeCfg(
        run="qwen",
        model_dir="/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        trust_remote_code=True,
        local_files_only=True,
    ),
    "deepseek": RuntimeCfg(
        run="deepseek",
        model_dir="/root/autodl-tmp/model_cache/deepseek-ai/deepseek-llm-7b-chat",
        trust_remote_code=True,
        local_files_only=True,
    ),
    "llama": RuntimeCfg(
        run="llama",
        model_dir="/root/autodl-tmp/demo/base/model_cache/LLM-Research/Meta-Llama-3-8B-Instruct",
        trust_remote_code=False,
        local_files_only=True,
    ),
}


def get_cfg(run: str) -> RuntimeCfg:
    """Return runtime configuration for the selected model."""
    run = run.strip().lower()

    if run not in MODEL_REGISTRY:
        raise ValueError(
            f"RUN must be one of {list(MODEL_REGISTRY.keys())}, got: {run}"
        )

    return MODEL_REGISTRY[run]


# =========================
# Smart Truncation for Pair Inputs
# =========================

PAIR_SEP = "\n\n-----\n\n"
CHUNK_HEAD_RE = re.compile(r"(\[Chunk[^\]]*\]\n)", re.M)


def _truncate_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return text

    token_ids = token_ids[:max_tokens]

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def to_text(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, str]:
    """
    Convert a chat-style JSONL sample into a single training text.

    If the user message contains two segment chunks separated by PAIR_SEP,
    this function applies balanced truncation to both chunks so that the
    final chat-template text fits within max_length.

    If the expected pair structure is not detected, the original chat
    template is used without special truncation.
    """
    messages = [dict(message) for message in example["messages"]]

    user_idx = next(
        (idx for idx, message in enumerate(messages) if message["role"] == "user"),
        None,
    )

    if user_idx is None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    user_content = messages[user_idx]["content"]

    if PAIR_SEP not in user_content:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    left, right = user_content.split(PAIR_SEP, 1)

    match_left = CHUNK_HEAD_RE.search(left)
    match_right = CHUNK_HEAD_RE.search(right)

    if not match_left or not match_right:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    prefix_left = left[:match_left.start()]
    head_a = match_left.group(1)
    body_a = left[match_left.end():].strip()

    head_b = match_right.group(1)
    body_b = right[match_right.end():].strip()

    # Estimate the fixed token overhead, including instructions,
    # chunk headers, chat template tokens, and the assistant answer.
    base_messages = [dict(message) for message in messages]
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

    body_a_cut = _truncate_tokens(tokenizer, body_a, each_budget)
    body_b_cut = _truncate_tokens(tokenizer, body_b, each_budget)

    new_user = (
        (prefix_left + head_a + body_a_cut).rstrip()
        + PAIR_SEP
        + (head_b + body_b_cut).strip()
    )
    messages[user_idx]["content"] = new_user

    final_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": final_text}


def get_lora_targets(run: str) -> List[str]:
    """Return LoRA target modules for causal language models."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]


def resolve_pair_files(data_dir: Path, a: int, b: int) -> Tuple[Path, Path]:
    """
    Resolve train and validation files for one segment pair.

    The function first looks for zero-padded names such as train_00-01.jsonl.
    If they do not exist, it falls back to non-padded names such as train0-1.jsonl.
    """
    padded_pair = f"{a:02d}-{b:02d}"

    train_file = data_dir / f"train_{padded_pair}.jsonl"
    val_file = data_dir / f"val_{padded_pair}.jsonl"

    if train_file.exists() and val_file.exists():
        return train_file, val_file

    train_file = data_dir / f"train{a}-{b}.jsonl"
    val_file = data_dir / f"val{a}-{b}.jsonl"

    return train_file, val_file


def train_one_pair(cfg: RuntimeCfg, a: int, b: int) -> None:
    """Fine-tune the model on one pair of physical-position segments."""
    train_file, val_file = resolve_pair_files(DATA_DIR, a, b)

    if not train_file.exists():
        print(f"Missing training file: {train_file}")
        return

    if not val_file.exists():
        print(f"Missing validation file: {val_file}")
        return

    out_dir = OUTPUT_ROOT / f"pair{a:02d}-{b:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Run: {cfg.run}")
    print(f"Pair: {a:02d}-{b:02d}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_dir,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

    dataset = dataset.map(
        to_text,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": MAX_SEQ_LENGTH,
        },
        remove_columns=dataset["train"].column_names,
        desc="Applying chat template and balanced truncation",
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_dir,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=get_lora_targets(cfg.run),
    )

    training_args = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,

        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=50,

        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"Finished pair {a:02d}-{b:02d}.")
    print(f"Adapter saved to: {adapter_dir}")

    del trainer, model, dataset, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    cfg = get_cfg(RUN)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    for a, b in PAIRS:
        if SKIP_DONE:
            done_dir = OUTPUT_ROOT / f"pair{a:02d}-{b:02d}" / "adapter"

            if done_dir.exists():
                print(f"Pair {a:02d}-{b:02d} already exists. Skipping.")
                continue

        train_one_pair(cfg, a, b)


if __name__ == "__main__":
    main()