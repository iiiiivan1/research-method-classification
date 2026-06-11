#!/bin/bash

# Example evaluation commands for representative models.

set -e

# BERT P4 evaluation
PAIR_ID=00-01 EVAL_BS=64 python src/evaluation/evaluate_bert.py

# SciBERT-long P4 evaluation
PAIR_ID=00-01 EVAL_BS=2 TRUNC_MODE=budget python src/evaluation/evaluate_scibertlong.py

# Qwen LLM evaluation
python src/evaluation/evaluate_llm.py