# Research Method Classification from Paper Segment Combinations
This repository provides the core implementation and sample data format for the paper:Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science
The project investigates how different physical-position segments of academic papers contribute to automatic multi-label research method classification.

## Overview
Research methods are important metadata for scholarly retrieval, literature review, and research intelligence analysis. However, research methods are often distributed across different parts of a paper rather than appearing only in a clearly named methodology section.
This project studies whether physical-position segments of full-text papers can help identify research methods. The body text of each paper is divided into ten approximately equal segments, and different segment-combination strategies are evaluated for multi-label classification.

## Task
The task is formulated as a multi-label text classification problem. Given the title, abstract, and selected body-text segments of an academic paper, the model predicts one or more research method labels from a predefined label set.
The label set contains 16 research method categories:
- Bibliometrics
- Content Analysis
- Delphi Study
- Ethnography / Field Study
- Experiment
- Focus Group
- Historical Method
- Interview
- Observation
- Questionnaire
- Research Diary / Journal
- Theoretical Approach
- Think Aloud Protocol
- Transaction Log Analysis
- Webometrics
- Other

## Segment Combination Schemes
The project uses several input construction strategies:
| Scheme | Description ||---|---|
| TA | Title + Abstract |
| C1 | A single body-text segment |
| C2 | Title + Abstract + a single body-text segment |
| C3 | Two body-text segments |
| C4 | Title + Abstract + two body-text segments |
The released code mainly provides representative implementations for P4-style pair inputs, which combine Title + Abstract with two body-text segments.

## Repository Structure
research-method-classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── label_set.json
│   └── sample_train.jsonl
│
├── prompts/
│   ├── method_summary_prompt.md
│   └── classification_prompt.md
│
├── src/
│   ├── preprocess/
│   │   ├── build_llm_base.py
│   │   ├── build_llm_segments.py
│   │   ├── build_bert_base.py
│   │   └── build_combinations.py
│   │
│   ├── models/
│   │   ├── train_bert.py
│   │   ├── train_scibertlong.py
│   │   └── train_qwen.py
│   │
│   └── evaluation/
│       ├── evaluate_bert.py
│       ├── evaluate_scibertlong.py
│       ├── evaluate_llm.py
│       └── metrics.py
│
└── scripts/
    ├── run_bert.sh
    ├── run_scibertlong.sh
    ├── run_qwen_gen.sh
    ├── run_qwen_eval.sh
    └── run_eval_examples.sh

## Data Availability
The full-text academic papers used in this study are not released due to copyright restrictions. This repository only provides the expected data format and sample files.Users should prepare their own full-text data and convert it into the required JSONL format before running the scripts.For details about the data format, see:text data/README.md 

## Models
This repository provides representative implementations for three model families:
| Model Family | Released Implementation |
|---|---|
| Short-context encoder | BERT |
| Long-context encoder | SciBERT-long |
| Large language model | Qwen |

Other models reported in the paper can be reproduced by replacing the model backbone while keeping the same data format, segment-combination strategy, and evaluation protocol.

## Evaluation
Encoder-based models and LLM-based models use different prediction formats.Encoder models output probability scores for each label. These scores are decoded using threshold, top-k, or hybrid strategies selected on the validation set.LLM-based models generate label strings directly. The generated labels are normalized and mapped to the predefined label set before metric calculation.All models are evaluated using multi-label classification metrics, including Micro-Precision, Micro-Recall, and Micro-F1.

## Installation
Install the required packages with:bash pip install -r requirements.txt 

## Example Usage
Run BERT evaluation for one pair:bash PAIR_ID=00-01 python src/evaluation/evaluate_bert.py 
Run SciBERT-long evaluation for one pair:bash PAIR_ID=00-01 EVAL_BS=2 python src/evaluation/evaluate_scibertlong.py 
Generate Qwen predictions:bash python src/evaluation/evaluate_llm.py 
Evaluate generated Qwen predictions:bash python src/evaluation/evaluate_llm.py 
For more examples, see:text scripts/ 

## Notes
- Full-text data are not included because of copyright restrictions.
- Trained checkpoints and large model files are not included.
- Local paths in the scripts may need to be modified according to the user's environment.
- The released code is intended to document the main experimental pipeline and support partial reproduction.

## Citation
If you use this code, please cite the corresponding paper:text Fang, Q., Hao, J., & Zhang, C. (2026). Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science. 

## Contact
For questions about the data or reproduction details, please contact the authors.