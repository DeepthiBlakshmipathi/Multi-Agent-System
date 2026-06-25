[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JG5U42VD)
# MAS for Automatic Survey

**Submission CSV header (exact):**

```
question,distribution,supports
```

- **`distribution`** — JSON object (single CSV cell) mapping the **exact option strings** to probabilities that **sum to 1.0 ± 1e-6**.  
- **`supports`** — JSON array (single CSV cell) of document IDs.

> **Note:** On the full competition you must output **exactly 100 unique** supports per question (deduplicated, deterministic). On the mini set you may have fewer.

---

## Quick start (Python 3.9, CPU)

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config.yaml` if needed, then:

```bash
# 1) Build (validate / prepare artifacts)
python -m index.build --config config.yaml --api_key dummy_api

# 2) Run (write the CSV submission)
python -m mas_survey.run --config config.yaml --api_key dummy_api
```

**Output:** `./artifacts/submission_js.csv`

---

## File layout

```
.
├── data/
│   ├── mini_documents.jsonl
│   └── dev/mini_dev.json
├── config.yaml
├── requirements.txt
├── index/
│   └── build.py
└── mas_survey/
    └── run.py
```

You provide the data files (mini or full):

* `data/mini_documents.jsonl` — one JSON per line with fields like `"id"`, `"title"`, `"description"`, `"post_content"`, `"content"`, …
* `data/dev/mini_dev.json` — a single JSON object keyed by the **full question string**; each value has a `distribution` stub (option keys with zeros) and an empty `supports` list.
* Full `documents.jsonl` (and embeddings) are on the Kaggle dataset;  
  they are **too large for this repo**.  
  **Do not commit large artifacts** (keep repo size <10 MB).  
  Always add them to `.gitignore`.

---

## CSV schema reminder

* Header **exactly**: `question,distribution,supports`
* UTF-8; follow CSV quoting (double quotes inside a quoted cell are doubled).
* Probabilities **≥ 0** and sum to **1.0 ± 1e-6**.

---

## FAQ

**Can I use GPUs?**
No. CPU-only.

**Can I add libraries?**
Yes, within course constraints. Start here, then add BM25 / FAISS / scikit-learn as needed.



## How to Run the Pipeline and What You’ll Get

This project implements a multi-agent system (MAS) for automatic survey response generation. It supports both mini and full datasets, and produces a submission-ready CSV file.

## Setup (Python 3.9, CPU-only)

```bash
python -m venv .venv
```
# On Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

Step:1 Build Indexes:
```bash
python -m index.build --config config.yaml --api_key dummy_api
```
Output:
Sparse index → index/sparse_index
Dense index → index/dense_index

This will:
Build a Whoosh sparse index from mini_documents.jsonl
Build a FAISS dense index from id_to_embedding.npz
Save both indexes in the index/ folder

Step:2 Run the survey pipeline: 
run.py with config.yaml: - 
```bash
python -m mas_survey.run --config config.yaml --api_key dummy_api
```
where You will see:
* Select which MAS pipeline to run:
1 or a → run main pipeline (run.py)
2 or b → run alternative pipeline 1 (run_1.py)
3 or c → run alternative pipeline 2 (run_2.py)
4 or d → exit
Enter your choice:

when choice is 1 or a :
Output :
* Loads the index artifacts built in Step 1
* Processes all questions listed in config.yaml
* Retrieves top-k documents using sparse and dense methods
* Reranks results using a CPU-only HF model (<0.8B)
* Outputs predictions to artifacts/submission_dev.csv
* Accepts --api_key even if unused

when choice is 2 or b :
run_1.py with config_1.yaml: - 
```bash
python -m mas_survey.run_1 --config config_1.yaml --api_key dummy_api
```
Output: 
Submission CSV  → artifacts/submission_js.csv
* Successfully processed all questions 
* Output saved to: artifacts/submission_js.csv 
* Distribution normalized to sum 1.0 
* Supports sliced from top-k results

when choice is 3 or c :
run_2.py with config_2.yaml: - 
```bash
python -m mas_survey.run_2 --config config_2.yaml --api_key dummy_api
```
Output:
Submission CSV → artifacts/mini_submission.csv
Final supports count: 100 (rubric-compliant) 
Output saved to: artifacts/mini_submission.csv 
Pipeline completed in 0.03 seconds 
Uses temperature and alpha for reranking 
Ensures exactly 100 unique document supports per question

for choice 4 or d it exits from the loop.

Where Each row contains:
question: full question string
distribution: JSON object mapping options to probabilities (sum = 1.0 ± 1e-6)
supports: JSON array of up to 100 unique document IDs

All outputs are reproducible with fixed seeds and run in under 3 minutes on CPU-only Python 3.9.

