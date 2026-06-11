This folder contains example shell scripts for running representative training and evaluation workflows.

| File | Description ||---|---|
| run_bert.sh | Runs BERT-based P4 training and evaluation for a selected segment pair. |
| run_scibertlong.sh | Runs SciBERT-long P4 training and evaluation with long-context input processing. |
| run_qwen_gen.sh | Generates Qwen predictions for validation and test sets. |
| run_qwen_eval.sh | Evaluates generated Qwen predictions using normalized label matching and multi-label metrics. |
| run_eval_examples.sh | Provides example commands for evaluating BERT, SciBERT-long, and Qwen models. |

These scripts are intended as lightweight examples. Users may need to modify paths, pair IDs, model directories, and batch sizes according to their local environment.