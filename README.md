# Research Method Classification 

> **Core implementation and sample data format for automatic research method classification from academic paper segments.**

This repository provides the core code and data format for the paper:  
*“Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science”*

The project investigates how different physical-position segments of academic papers contribute to automatic multi-label research method classification.

---

##  Overview & Task

Research methods are essential metadata for scholarly retrieval, literature reviews, and research intelligence analysis. However, these methods are often distributed across various parts of a paper rather than neatly contained in a clearly named "Methodology" section.

This project evaluates whether the physical-position segments of full-text papers can help identify these methods. The body text of each paper is divided into ten approximately equal segments, and different segment-combination strategies are evaluated for a **multi-label text classification task**.

Given the title, abstract, and selected body-text segments of an academic paper, the model predicts one or more research method labels from the following predefined set:

### Research Method Labels

| No. | Method Label | No. | Method Label |
|---:|---|---:|---|
| **1** | Bibliometrics | **9** | Observation |
| **2** | Content Analysis | **10** | Questionnaire |
| **3** | Delphi Study | **11** | Research Diary / Journal |
| **4** | Ethnography / Field Study | **12** | Theoretical Approach |
| **5** | Experiment | **13** | Think Aloud Protocol |
| **6** | Focus Group | **14** | Transaction Log Analysis |
| **7** | Historical Method | **15** | Webometrics |
| **8** | Interview | **16** | Other |

---

## 📁 Repository Structure

```text
research-method-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── label_set.json
│   └── sample_train.jsonl
├── prompts/
│   ├── method_summary_prompt.md
│   └── classification_prompt.md
├── src/
│   ├── preprocess/
│   ├── models/
│   └── evaluation/
└── scripts/
    ├── run_bert.sh
    ├── run_scibertlong.sh
    ├── run_qwen_gen.sh
    ├── run_qwen_eval.sh
    └── run_eval_examples.sh
 ```
 ## ⚠️ Data Availability

> **Note:** The full-text academic papers used in this study are **not released** due to copyright restrictions.

This repository only provides the expected data format and sample files. Users should prepare their own full-text data and convert it into the required JSONL format before running the scripts.

For detailed information about the required data format, see `data/README.md`.

---

##  Models & Evaluation

We provide representative implementations for three model families. Encoder outputs are decoded using threshold, top-k, or hybrid strategies (selected on the validation set). LLM outputs are normalized and mapped to the predefined label set before metric calculation.

| Model Family | Released Implementation | Prediction Format |
|---|---|---|
| **Short-context encoder** | BERT | Probability scores for each label |
| **Long-context encoder** | SciBERT-long | Probability scores for each label |
| **Large language model** | Qwen | Generated label strings |

**Evaluation Metrics:**
All models are evaluated using **Micro-Precision**, **Micro-Recall**, and **Micro-F1**.

---

## Installation

Install the required packages with pip:

```bash
pip install -r requirements.txt
```
##  Example Usage

Run BERT evaluation for one pair:
```bash
PAIR_ID=00-01 python src/evaluation/evaluate_bert.py 
```

Run SciBERT-long evaluation for one pair:
```bash
PAIR_ID=00-01 EVAL_BS=2 python src/evaluation/evaluate_scibertlong.py 
```

Generate and evaluate Qwen predictions:
```bash
python src/evaluation/evaluate_llm.py 
```

*(For more examples, see the `scripts/` folder.)*

---

##  Notes

* **Copyright Limitations:** Full-text data, trained checkpoints, and large model files are not included.
* **Environment Setup:** Local paths in the scripts may need to be modified according to your specific environment.
* **Purpose:** The released code is intended to document the main experimental pipeline and support partial reproduction of the paper's findings.

---

##  Citation & Contact
* For questions about the data or reproduction details, please contact the authors.
* If you use this code or data format in your research, please cite our paper (accepted at **ASIS&T 2026**):

> Fang, Q., Hao, J., & Zhang, C. (2026). Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science. ***Proceedings of the 89th Annual Meeting of the Association for Information Science and Technology (ASIST’2026)***, Bangkok, Thailand, 6-10 November, 2026. [[doi]]  [[arXiv]]()  [[Dataset & Source Code]](https://github.com/iiiiivan1/research-method-classification)



