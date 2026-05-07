# An Experiment with LLM in Contract Extraction

Author: Nhan Le | Supervisor: Son Tran

---

## Motivation

Supply chains are multi-agent networks involving production, delivery, and payments. Contracts encode obligations who does what, by when, but manual formalization is slow and error-prone, hindering automated monitoring, diagnosis, and planning.

**Goal:** Automatically extract formal clauses from natural-language contracts so agents can reason and act on them.

The extracted clauses are designed to feed into a multi-agent contract monitoring system capable of reasoning, planning, diagnosis, and explanation.

---

## Key Contributions

- A practical pipeline to translate natural-language contracts into formal clauses using LLM-based extraction and DSPy-based verification.
- A small corpus of 11 synthetic supply-chain contracts + tests on 123 real-world NDA contracts (ContractNLI dataset).
- Insights on where Chain-of-Thought verification helps (entity verification) and where it struggles (full clause entailment).
- An integration path to a supply-chain monitoring agent architecture.

---

## Clause Format

All pipelines generate clauses grounded in the **ℒc contract specification language**, using this standardized form:

```
C<n>: <party> is responsible for <action/obligation> when <time_expression>
```

Where `time_expression` can be: `always`, `eventually`, `per_unit[n...m]`, or `by_unit n`.

**Example:**
```
C1: Lumber Yard A is responsible for producing Number 2 Common grade lumber (144,000 board feet) when by week 4.
C2: Lumber Yard A is responsible for delivering lumber to XYZ Homes when by week 4.
C3: XYZ Homes is responsible for making payment ($122,000) for the delivered lumber when by week 4.
```

---

## Method Overview

The pipeline runs in two phases:

**Phase 1 — LLM Extraction**
1. `ContractAnalysis`: extracts parties, obligations, timelines, and financial terms
2. `ClauseGeneration`: synthesizes clauses in canonical format

**Phase 2 — DSPy Verification**
3. `EntailmentVerification` — verifies each clause is logically entailed by the original contract text

### Design Choice: Predict vs. ChainOfThought

- **Steps 1–2 use `dspy.Predict`** Chain-of-Thought produces verbose preambles that pollute the strict clause format and break structured parsing.
- **Step 3 uses `dspy.ChainOfThought`** Explicit reasoning provides better factual grounding and evidence snippets for entailment judgments.
- **Post-filter:** only clauses labeled `ENTAILMENT` (without "NOT") are kept.

---

## Pipelines

| Folder | Model | Status | Description |
|--------|-------|--------|-------------|
| `main/` | GPT-4o-mini + DSPy 2.4.5 | **Active** | Full clause generation + entailment verification on ContractNLI dataset |
| `test_LLMA3 through Olama/` | LLaMA 3 (Ollama local) + DSPy | Archive | Local LLM pipeline with DSPy BootstrapFewShot verifier |
| `test_ChatGPT : Gemini/` | GPT-3.5-turbo / Gemini 1.0 Pro | Archive | Original proof-of-concept on 3 hand-crafted contracts |

---

## Results

The pipeline was tested with both local and cloud-based LLMs via DSPy's unified interface:

- **Local models** (llama3.2, deepseek-r1:8b, qwen3:8b) inconsistent adherence to the required clause format; frequent reasoning verbosity interfered with structured parsing.
- **GPT-4o-mini**  superior performance in maintaining strict clause templates and more reliable entailment judgments. ← recommended
- **Modular design** the underlying model can be swapped without changing pipeline logic, making future model upgrades straightforward.

---

## Quick Start (main — recommended)

1. Set up a Python 3.12 virtual environment and install dependencies (see `main/README.md`).
2. Set your OpenAI API key.
3. Run:

```bash
cd main
python improved_clause_generator.py
```

See `main/README.md` for full setup and configuration details.

---

## Project Structure

```
A Experiment with LLM in Contract Extraction/
├── README.md                              ← this file
│
├── main/                                  ← MAIN: DSPy + GPT-4o-mini
│   ├── README.md
│   ├── improved_clause_generator.py
│   ├── scriptJson.py
│   ├── data/
│   │   ├── test.json
│   │   ├── test_11contracts.json
│   │   ├── test_modified.json
│   │   ├── train.json
│   │   └── train_modified.json
│   └── output/
│       ├── generated_clauses_output.json
│       └── generated_clauses_output_NLI.json
│
├── test_LLMA3 through Olama/              ← LLaMA 3 (Ollama) + DSPy (archive)
│   ├── README.md
│   ├── llama3.py
│   ├── testingClause.py
│   ├── data/
│   │   └── dev_data.json
│   └── output/
│       ├── output_json.json
│       └── compiled_contract_details.joblib
│
└── test_ChatGPT : Gemini/                 ← GPT-3.5-turbo / Gemini (earliest, archive)
    ├── README.md
    ├── chatGPT.py
    ├── gemini_model.py
    ├── contractData.py
    └── contractData.json
```

---

## Limitations

- Implicit obligations and industry custom are not captured.
- Cross-clause dependencies (conditionals) are not checked jointly.
- Temporal complexity (recurrence, triggers) is simplified to week units.
- English-only; no multilingual support.

---

## Future Work

- Add rule-based and symbolic checks for temporal and logical consistency.
- Build graph-level verification across clause sets.
- Expand evaluation with more real-world contracts.
- Add multilingual support and domain ontologies to capture implicit duties.

---

## Dataset

Source: **ContractNLI** (Koreeda & Manning, 2021) — included in `main/data/`

- `train.json` — 7,600+ contracts (training)
- `test.json` — test split
- `test_modified.json` / `train_modified.json` — cleaned versions generated by `main/scriptJson.py`

---

## References

1. Y. Koreeda, C. Manning, *ContractNLI: A dataset for document-level natural language inference for contracts*, 2021.
2. D. Flynn, C. Nadeau, J. Shantz, M. Balduccini, T. C. Son, E. Griffor, *Formalizing and Reasoning about Supply Chain Contracts between Agents*, 25th PADL, vol. 13880, 2023.
3. L. Tran, T. C. Son, D. Flynn, M. Balduccini, *A Multi-Agent Simulation for Supply Chains Contract Execution*, Technical Report, NMSU, 2024.
4. M. Gelfond, V. Lifschitz, *Action Languages*, Electronic Transactions on Artificial Intelligence 3, 1998.
