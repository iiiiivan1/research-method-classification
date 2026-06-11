#!/bin/bash

# Generate Qwen predictions for validation and test sets.

set -e

# Edit MODE, START_PAIR, and END_PAIR in src/evaluation/evaluate_llm.py if needed.
# MODE should be set to "gen" before running this script.

python src/evaluation/evaluate_llm.py