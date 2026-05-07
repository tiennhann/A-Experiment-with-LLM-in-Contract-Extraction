# test_ChatGPT : Gemini — Earliest Experiments

> **Archive folder.** This was the original proof-of-concept. The active pipeline lives in `main/`.

Uses **OpenAI GPT-3.5-turbo** and **Google Gemini 1.0 Pro** to extract contract details, generate clauses, and evaluate entailment on three hand-crafted sample contracts.

---

## Overview

Both scripts follow the same three-prompt pattern:

```
contractData.py  (3 sample contracts: XYZ Homes, DEF Tech, GHI Restaurants)
    │
    ▼  Prompt 1
    generate_contract_details()   → extracts parties, product, quantity, grade, timeline, cost
    │
    ▼  Prompt 2
    get_clauses()                 → formats as: C<n>: party responsible for ... when ...
    │
    ▼  Prompt 3
    evaluate_entailment()         → checks each clause against original contract text
    │
    ▼  Output
    prints entailed clauses to stdout
```

---

## Folder Structure

```
test_ChatGPT : Gemini/
├── README.md
├── chatGPT.py        # GPT-3.5-turbo version (OpenAI API)
├── gemini_model.py   # Gemini 1.0 Pro version (Google Generative AI)
├── contractData.py   # Shared: defines contract_1, contract_2, contract_3 strings
└── contractData.json # Same contracts in JSON format (used by chatGPT.py)
```

---

## Requirements

- Python 3.12
- OpenAI API key (for `chatGPT.py`)
- Google Generative AI API key (for `gemini_model.py`)

### Key dependencies

| Package | Purpose |
|---------|---------|
| `openai 1.31.0` | ChatGPT API client |
| `google-generativeai` | Gemini API client |
| `python-dotenv` | Load API keys from `.env` |

Install dependencies:

```bash
pip install openai==1.31.0 google-generativeai python-dotenv
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Set API keys

**For `chatGPT.py`** — set as an environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or update line 12 in `chatGPT.py` directly:

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
```

**For `gemini_model.py`** — update line 10:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")
```

---

## Running

### ChatGPT version (GPT-3.5-turbo)

Run from the `test_ChatGPT : Gemini/` directory:

```bash
python chatGPT.py
```

Reads `contractData.json`, prints extracted details, generated clauses, and entailment results to stdout.

### Gemini version (Gemini 1.0 Pro)

```bash
python gemini_model.py
```

Uses `contract_1`, `contract_2`, `contract_3` imported directly from `contractData.py`.

---

## Sample Contracts (`contractData.py`)

| Contract | Parties | Product |
|----------|---------|---------|
| `contract_1` | XYZ Homes & Lumber Yard A | 144,000 board feet Number 2 Common lumber |
| `contract_2` | DEF Tech & Semiconductor Chip Supplier C | 10,000 semiconductor chips |
| `contract_3` | GHI Restaurants & Fresh Produce Supplier D | 40,000 lbs fresh produce |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AuthenticationError` | Check your API key in the script or environment |
| `ModuleNotFoundError: google.generativeai` | Run `pip install google-generativeai` |
| `ModuleNotFoundError: contractData` | Run the script from the `test_ChatGPT : Gemini/` folder |
| Rate limit errors | Add `time.sleep()` between API calls |
