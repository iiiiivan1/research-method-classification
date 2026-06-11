#!/bin/bash

set -e

export PAIR_ID=00-01
export EVAL_BS=64
export TA_MAX_TOKENS=160

python src/models/train_bert.py
python src/evaluation/evaluate_bert.py