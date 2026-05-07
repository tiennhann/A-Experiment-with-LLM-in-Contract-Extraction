# test_LLMA3 through Olama — LLaMA 3 via Ollama + DSPy Clause Verifier

> **Archive folder.** This was an intermediate experiment with a local LLM. The active pipeline lives in `main/`.

Generates contract clauses using a **local LLaMA 3 model via Ollama**, then trains a **DSPy BootstrapFewShot** model to verify whether each clause is entailed by the original contract.

---

## Overview

This pipeline has two stages:

```
dev_data.json
    │
    ▼  Stage 1: llama3.py
    LLaMA 3 (Ollama local)
    ├── generate_contract_details()   → extracts parties, quantities, grades, costs
    ├── get_clauses()                 → formats clauses as C<n>: party responsible for ... when ...
    └── evaluate_entailment()         → keeps only entailed clauses
    │
    ▼
output/output_json.json               (training data for Stage 2)
    │
    ▼  Stage 2: testingClause.py
    DSPy BootstrapFewShot
    ├── trains GenerateContractDetails module on output_json.json
    └── saves compiled model → output/compiled_contract_details.joblib
```

---

## Folder Structure

```
test_LLMA3 through Olama/
├── README.md
├── llama3.py            # Stage 1 — LLaMA 3 clause generator + entailment filter
├── testingClause.py     # Stage 2 — DSPy clause verifier (train / run)
├── data/
│   └── dev_data.json    # Input contracts (dev subset)
└── output/
    ├── output_json.json                  # Stage 1 output / Stage 2 training data
    └── compiled_contract_details.joblib  # Saved DSPy model
```

---

## Requirements

- Python 3.12
- **Ollama** running locally with LLaMA 3 pulled

### Key dependencies

| Package | Purpose |
|---------|---------|
| `dspy-ai 2.4.5` | BootstrapFewShot training |
| `requests` | Ollama HTTP API calls |
| `joblib` | Model save/load |
| `scikit-learn` | TF-IDF semantic similarity metric |

Install dependencies:

```bash
pip install dspy-ai==2.4.5 requests joblib scikit-learn
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install dspy-ai==2.4.5 requests joblib scikit-learn
```

### 3. Start Ollama and pull LLaMA 3

```bash
ollama serve
ollama pull llama3
```

---

## Running

### Stage 1 — Generate clauses from contracts

Run from the `test_LLMA3 through Olama/` directory:

```bash
python llama3.py
```

Reads `data/dev_data.json`, outputs `output/output_json.json`.

### Stage 2 — Train DSPy verifier and run on a test contract

```bash
python testingClause.py
```

- If `output/compiled_contract_details.joblib` exists → loads pre-trained model and runs inference
- If not → trains on `output/output_json.json`, saves model, then runs inference

---

## Input Format (`dev_data.json`)

```json
{
  "documents": [
    {
      "file_name": "contract1",
      "text": "Contract text here..."
    }
  ]
}
```

## Output Format (`output_json.json`)

```json
{
  "contracts": [
    {
      "contract_text": "...",
      "question": "Is the clause entailed with the given contract_text?",
      "clauses": [
        {"clause": "C1: Lumber Yard A responsible for ...", "answer": "yes"}
      ]
    }
  ]
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` first |
| `model not found` | Run `ollama pull llama3` |
| `ModuleNotFoundError` | Activate your virtual environment and install dependencies |
| Model file corrupt | Delete `output/compiled_contract_details.joblib` and retrain |
