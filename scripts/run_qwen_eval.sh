#!/bin/bash

# Evaluate generated Qwen predictions.

set -e

# Edit MODE, START_PAIR, and END_PAIR in src/evaluation/evaluate_llm.py if needed.
# MODE should be set to "eval" before running this script.

python src/evaluation/evaluate_llm.py