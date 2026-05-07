# main — Contract Clause Generator

Automatically extracts, generates, and verifies formal clauses from supply-chain contracts using **DSPy 2.4.5** and **GPT-4o-mini** (or local Ollama).

---

## Project Overview

This pipeline takes a contract text and produces structured, verifiable clauses in the format:

```
C1: Lumber Yard A is responsible for producing Number 2 Common grade lumber (144,000 board feet) when by week 4.
C2: XYZ Homes is responsible for making payment ($122,000) for the delivered lumber when by week 4.
```

### Three-Step Process

```
Contract Text
    │
    ▼  Step 1
ContractAnalysis          → Extracts parties, obligations, timelines, financial terms
    │
    ▼  Step 2
ClauseGeneration          → Generates formal C<n>: <party> responsible for <action> when <deadline> clauses
    │
    ▼  Step 3
EntailmentVerification    → Verifies each clause is logically entailed by the original contract
    │
    ▼
output/generated_clauses_output_NLI.json
```

---

## Folder Structure

```
main/
├── README.md
├── improved_clause_generator.py   # Main script
├── scriptJson.py                  # Preprocessor: strips raw ContractNLI JSON
├── data/
│   ├── test.json                  # Raw ContractNLI test set
│   ├── test_11contracts.json      # Small test set (11 hand-crafted contracts)
│   ├── test_modified.json         # Full ContractNLI test set (cleaned)
│   ├── train.json                 # Raw ContractNLI train set
│   └── train_modified.json        # Full ContractNLI train set (cleaned)
└── output/
    ├── generated_clauses_output.json        # Output from test_11contracts run
    └── generated_clauses_output_NLI.json    # Output from full test set run
```

---

## Requirements

- Python 3.12
- OpenAI API key (for GPT-4o-mini) or Ollama running locally

### Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `dspy-ai` | 2.4.5 | LLM orchestration framework |
| `openai` | 1.31.0 | GPT-4o-mini API client |
| `scikit-learn` | 1.5.1 | TF-IDF semantic similarity scoring |
| `python-dotenv` | 1.0.1 | Environment variable loading |

Install all dependencies:

```bash
pip install dspy-ai==2.4.5 openai==1.31.0 scikit-learn==1.5.1 python-dotenv==1.0.1
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
pip install dspy-ai==2.4.5 openai==1.31.0 scikit-learn==1.5.1 python-dotenv==1.0.1
```

### 3. Set your OpenAI API key

Option A — environment variable (recommended):

```bash
export OPENAI_API_KEY="your-key-here"
```

Option B — edit `improved_clause_generator.py` directly and replace the placeholder on line 23:

```python
turbo = dspy.OpenAI(
    model='gpt-4o-mini',
    api_key="YOUR_OPENAI_API_KEY_HERE",   # <-- replace this
    max_tokens=3000
)
```

### 4. (Optional) Use local Ollama instead of OpenAI

If you want to run fully offline, swap the configured model in `improved_clause_generator.py`:

```python
# Comment out OpenAI:
# dspy.settings.configure(lm=turbo)

# Uncomment Ollama:
dspy.settings.configure(lm=ollama_model)   # uses qwen3:8b by default
```

Make sure Ollama is running:

```bash
ollama serve
ollama pull qwen3:8b   # or: ollama pull llama3.2
```

---

## Running

### Quick test (11 contracts — recommended first run)

In `improved_clause_generator.py`, uncomment the small test paths and comment out the full test:

```python
# Small test (11 contracts) — faster to verify setup
test_input  = os.path.join(BASE_DIR, "data", "test_11contracts.json")
test_output = os.path.join(BASE_DIR, "output", "generated_clauses_output.json")

# Full ContractNLI test set  ← comment this out
# test_input  = os.path.join(BASE_DIR, "data", "test_modified.json")
# test_output = os.path.join(BASE_DIR, "output", "generated_clauses_output_NLI.json")
```

Then run from the `main/` directory:

```bash
python improved_clause_generator.py
```

### Full ContractNLI test set run

The default config in `main()` already points to the full dataset. Just run:

```bash
python improved_clause_generator.py
```

---

## Preprocessing Raw Data (only needed if re-generating from scratch)

If you need to regenerate `test_modified.json` / `train_modified.json` from the raw ContractNLI source files:

```bash
python scriptJson.py
```

This reads `data/test.json` and `data/train.json` and writes cleaned versions to `data/`.

---

## Output Format

Results are saved as JSON with this structure:

```json
{
  "contracts": [
    {
      "file_name": "contract_001",
      "contract_text": "...",
      "analysis": "1/ Parties and Roles: ...",
      "all_clauses": ["C1: ...", "C2: ..."],
      "verified_clauses": [
        {
          "clause": "C1: Lumber Yard A is responsible for ...",
          "reasoning": "The contract explicitly states ...",
          "verdict": "ENTAILMENT"
        }
      ],
      "rejected_clauses": [
        {
          "clause": "C3: ...",
          "verdict": "NOT_ENTAILMENT",
          "reasoning": "The amount mentioned does not match ..."
        }
      ],
      "question": "Is the clause entailed with the given contract_text?",
      "clauses": [{"clause": "C1: ...", "answer": "yes"}]
    }
  ]
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: dspy` | Activate your virtual environment and run `pip install dspy-ai==2.4.5` |
| `AuthenticationError: openai` | Set your `OPENAI_API_KEY` environment variable or update it in the script |
| `FileNotFoundError: test_modified.json` | Run `python scriptJson.py` first to generate the data file |
| Ollama connection error | Start Ollama with `ollama serve` before running |
| Rate limit errors (OpenAI) | The script has a 2-second delay between contracts; increase `time.sleep(2)` if needed |
