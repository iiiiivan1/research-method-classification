#!/bin/bash

# Run SciBERT-long P4 training and evaluation for one segment pair.

set -e

export PAIR_ID=00-01
export EVAL_BS=2
export TRUNC_MODE=budget
export MAX_TA=0
export MAX_I=0
export MAX_J=0

python src/models/train_scibertlong.py

python src/evaluation/evaluate_scibertlong.py