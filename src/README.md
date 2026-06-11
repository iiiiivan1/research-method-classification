This folder contains the core implementation of the project.The released code provides representative implementations for the main experimental pipeline, including data preprocessing, model training, and evaluation.

Module Description
 preprocess/This folder contains scripts for data preprocessing and input construction.

| File | Description ||---|---|
| build_segments.py | Parses raw TXT files, extracts fields such as title, abstract, full text, and method labels, and divides the body text into ten physical-position segments. |
| build_llm_data.py | Converts processed samples into chat-style JSONL files for supervised fine-tuning of LLMs. |

 models/This folder contains representative model implementations for three model families used in the study.

| File | Description ||---|---|
| train_bert.py | Implements a short-context encoder model for multi-label research method classification. |
| train_scibertlong.py | Implements a long-context encoder model for multi-label research method classification. |
| train_qwen_qlora.py | Implements LLM fine-tuning with Qwen using QLoRA. |

The three released scripts correspond to the following model families:
| Short-context encoder | BERT |
| Long-context encoder | SciBERT-long |
| Large language model | Qwen with QLoRA |

Other models reported in the paper follow the same data format, segment-combination strategy, and evaluation protocol.


evaluation/This folder contains scripts for prediction post-processing and evaluation.
| File | Description ||---|---|
| evaluate.py | Computes evaluation metrics such as Micro-Precision, Micro-Recall, and Micro-F1. |
| evaluate_llm.py | Parses LLM-generated labels and evaluates them against the gold labels. |
| label_mapping.py | Normalizes label names and maps model outputs to the predefined label set. |