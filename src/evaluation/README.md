This folder contains evaluation scripts for encoder-based models and LLM-based models.

| File | Description ||---|---|
| evaluate_encoder.py | Evaluates encoder-based models such as BERT and SciBERT-long. It loads trained LoRA adapters, predicts logits, converts logits into probabilities using Sigmoid, selects labels through validation-based decoding strategies, and reports multi-label classification metrics. |
| evaluate_llm.py | Evaluates LLM-based models such as Qwen. It supports generation and evaluation modes. Generated label strings are normalized and mapped to the predefined research method label set before metric calculation. |
| metrics.py | Provides shared metric functions for multi-label evaluation, including Micro-Precision, Micro-Recall, Micro-F1, per-label reports, and label normalization utilities. |

Encoder-based and LLM-based models use different prediction formats. Encoder models produce probability scores for each label, while LLMs generate label strings directly. To ensure comparability, both outputs are converted into the same normalized label format before calculating evaluation metrics.