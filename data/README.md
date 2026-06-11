Data
This folder describes the data format used in this project.

Data Availability
The full-text academic papers used in this study are not publicly released due to copyright restrictions. The original corpus was constructed from full-text articles in Library and Information Science journals and annotated research method labels.Only sample files are provided to illustrate the expected input and output formats. Researchers interested in full reproduction may contact the authors for more information.

Raw Data Format
The original data are stored as .txt files. Each file represents one academic paper and contains structured fields marked by tags.A simplified example is shown in original data.txt


Segmented Data Format
After preprocessing and linear position-based partitioning, the body text is divided into ten segments. Each segment represents approximately 10% of the full body text..The segmented data are stored in jsonl format. Each line represents one body-text segment.A sample record is shown in segment. jsonl.

Label Set
The classification task includes 16 research method categories:
[
  "Bibliometrics",
  "Content analysis",
  "Delphi study",
  "Ethnography/field study",
  "Experiment",
  "Focus groups",
  "Historical method",
  "Interview",
  "Observation",
  "Questionnaire",
  "Research diary/journal",
  "Theoretical approach",
  "Think-aloud protocol",
  "Transaction log analysis",
  "Webometrics",
  "Other methods"
]