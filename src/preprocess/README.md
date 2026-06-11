This folder contains scripts for raw data parsing, segment construction, and model-specific input generation.

| File | Description |
|---|---|
| build_llm_base.py | Constructs the Title + Abstract baseline data in chat-style JSONL format for LLM fine-tuning. |
| build_llm_segments.py | Constructs LLM input files based on single physical-position segments. The body text is divided into ten segments and converted into chat-style JSONL files. |
| build_bert_base.py | Constructs the Title + Abstract baseline data for encoder-based multi-label classification models such as BERT. |
| build_combinations.py | Constructs segment-combination inputs for encoder-based models, including dual-segment and TA-augmented dual-segment combinations. |