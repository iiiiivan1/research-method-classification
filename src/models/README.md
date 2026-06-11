This folder contains representative training scripts for the model families used in the study.

| File | Description ||---|---|
| train_bert.py | Trains a short-context BERT encoder for multi-label research method classification. The input is usually Title + Abstract or truncated segment-combination text. |
| train_scibertlong.py | Trains a long-context SciBERT-long encoder for multi-label research method classification. This script is used for longer segment-combination inputs. |
| train_qwen.py | Fine-tunes Qwen for LLM-based research method classification using instruction-style data. |

The three scripts correspond to three representative model families:
| Model Family | Released Implementation ||---|---|
| Short-context encoder | BERT |
| Long-context encoder | SciBERT-long |
| Large language model | Qwen |

Other models reported in the paper can be reproduced by replacing the model backbone while keeping the same data format, segment-combination strategy, and evaluation protocol.