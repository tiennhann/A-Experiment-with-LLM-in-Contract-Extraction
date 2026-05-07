"""
Improved General-Purpose Contract Clause Generator using DSPy 2.4.5
Supports all contract types: NDAs, employment, service agreements, supply chain, etc.
Author: Nhan Le 
"""

import dspy
import json
import os
from typing import List
from dspy.teleprompt import BootstrapFewShot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION - Update based on your setup (DSPy 2.4.5 compatible)
# ============================================================================

# Option 1: Use OpenAI
import openai
turbo = dspy.OpenAI(
    model='gpt-4o-mini', 
    api_key="YOUR_OPENAI_API_KEY_HERE", 
    max_tokens=3000)


# Option 2: Use local Ollama (current setup)
ollama_model = dspy.OllamaLocal(
    model="qwen3:8b",  # or llama3.2
    model_type='text',
    max_tokens=3000,
    temperature=0.1,
    top_p=0.9,
    frequency_penalty=0.0,
    top_k=50
)

#dspy.settings.configure(lm=ollama_model)
dspy.settings.configure(lm=turbo)

# ============================================================================
# DSPY SIGNATURES - Define the structure of the AI modules
# ============================================================================

class ContractAnalysis(dspy.Signature):
    """Analyze a contract and extract key structured information for all contract types."""
    
    contract_text = dspy.InputField(desc="The full contract text to analyze")
    analysis = dspy.OutputField(
        desc="""Extract and structure the contract into these sections (use EXACTLY these headers):

                1/ Parties and Roles:
                - Party 1: [name] - Role: [role]
                - Party 2: [name] - Role: [role]

                2/ Key Obligations:
                - [party_name]: [list specific obligations]
                - [party_name]: [list specific obligations]

                3/ Timeline and Deadlines (convert months to weeks: 1 month = 4 weeks):
                - [obligation]: [timeline in weeks]

                4/ Financial Terms:
                - Payment cost: [payment cost if any]
                - Transport and delivery cost: [shipping cost if any]
                - Other costs: [any other costs]

                5/ Special Conditions:
                - Prerequisites: [list conditions]
                - Contingencies: [list dependencies]

                6/ Termination Clauses:
                - Conditions for termination: [list conditions]
                - Notice period requirements: [specify period in weeks]

                Extract ONLY information explicitly stated in the contract."""
    )

class ClauseGeneration(dspy.Signature):
    """Generate formal, verifiable clauses from contract analysis."""
    
    contract_text = dspy.InputField(desc="The original contract text")
    analysis = dspy.InputField(desc="Structured analysis of the contract")
    clauses = dspy.OutputField(
        desc="""Generate clauses in this EXACT format ONLY (no explanatory text):

            Required Format: C<number>: <party> responsible for <action/obligation> when <timeline/condition>

            Requirements:
                1. Party names: Use natural spacing (e.g., Lumber Yard A, XYZ Homes, John Smith, TechCorp Inc)
                2. Actions: Be specific and descriptive in natural language
                    - For quantities: Use commas in numbers (e.g., 144,000 board feet, not 144000)
                    - For money: Use $ with commas (e.g., $122,000, not USD_122000)
                    - Include all relevant details (grade, specifications) in natural language
                3. Timeline: Convert to weeks (1 month = 4 weeks), use format like "by week 4" or "within 2 weeks"
                4. Start each line with C followed by a number (C1, C2, etc.)

            CRITICAL RULES:
                - Start EVERY clause with "C" followed by a number (C1:, C2:, C3:, etc.)
                - Always use the phrase "is responsible for" (not just "responsible for")
                - Use natural spacing in names (spaces, not underscores)
                - Use commas in large numbers: 144,000 not 144000
                - Use proper currency format: $122,000 not USD_122000
                - Convert all time periods to weeks (1 month = 4 weeks)
                - Write actions in natural, descriptive language


            Examples:
                C1: Lumber Yard A is responsible for producing Number 2 Common grade lumber (144,000 board feet) when by week 4.
                C2: Lumber Yard A is responsible for delivering Number 2 Common grade lumber (144,000 board feet) to XYZ Homes when by week 4.
                C3: XYZ Homes is responsible for making payment ($122,000) for the delivered lumber when by week 4.
                C4: Employee is responsible for maintaining confidentiality of company information when contract is active.
                C5: Employer is responsible for making salary payment ($10,000 monthly) when every 4 weeks.

            Generate clauses for ALL obligations, payments, timelines, and conditions mentioned in the contract.
            Output ONLY the clauses, one per line, no additional text."""
    )

class EntailmentVerification(dspy.Signature):
    """Verify if a clause is logically entailed by the contract text."""
    
    contract_text = dspy.InputField(desc="The original contract text")
    clause = dspy.InputField(desc="The clause to verify")
    is_entailed = dspy.OutputField(
        desc="""Determine if the clause is entailed by checking:

            1. Parties: All mentioned parties exist in contract
            2. Actions: The obligation/action is explicitly stated or clearly implied
            3. Amounts: Numbers, quantities, costs match exactly
            4. Timeline: Time periods are consistent (remember 1 month = 4 weeks)
            5. Conditions: No contradictions with other terms

            Answer ONLY with one word: 'ENTAILMENT' or 'NOT_ENTAILMENT'"""
    )
    reasoning = dspy.OutputField(desc="Brief 1-2 sentence explanation referencing specific contract text")

# ============================================================================
# DSPY MODULES - Combine signatures with reasoning strategies
# ============================================================================

class ContractClauseGenerator(dspy.Module):
    """Main module for generating and verifying contract clauses."""
    
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for step-by-step reasoning
        self.analyze_contract = dspy.Predict(ContractAnalysis)
        self.generate_clauses = dspy.Predict(ClauseGeneration)
        self.verify_entailment = dspy.ChainOfThought(EntailmentVerification)
    
    def forward(self, contract_text: str, verify: bool = True):
        """
        Process a contract to generate and optionally verify clauses.
        
        Args:
            contract_text: The full contract text
            verify: Whether to verify each clause for entailment
            
        Returns:
            dspy.Prediction with analysis, clauses, and verified_clauses
        """
        print("  Step 1: Analyzing contract structure...")
        # Step 1: Analyze the contract structure
        analysis_result = self.analyze_contract(contract_text=contract_text)
        # analysis = analysis_result.analysis
        
        print("  Step 2: Generating formal clauses...")
        # Step 2: Generate formal clauses
        clause_result = self.generate_clauses(
            contract_text=contract_text,
            analysis=analysis_result.analysis
        )
        
        # Remove <think> tags and everything in them
        import re
        clauses_text = clause_result.clauses
        clauses_text = re.sub(r'<think>.*?</think>', '', clauses_text, flags=re.DOTALL)
        clauses_text = clauses_text.strip()


        # DEBUG: Print raw output to see what model is generating
        print(f"\n  DEBUG - Raw model output:\n{clauses_text}\n")
        
        # Parse individual clauses
        clauses = [c.strip() for c in clauses_text.split('\n') if c.strip() and c.strip().startswith('C')]
        
        print(f"  Generated {len(clauses)} clauses")
        
        verified_clauses = []
        rejected_clauses = []
        
        if verify and clauses:
            print(f"  Step 3: Verifying entailment for {len(clauses)} clauses...")
            
            for clause in clauses:
                verification = self.verify_entailment(
                    contract_text=contract_text,
                    clause=clause
                )
                
                # Check if entailed
                verdict = verification.is_entailed.upper()
                if 'ENTAILMENT' in verdict and 'NOT' not in verdict:
                    verified_clauses.append({
                        'clause': clause,
                        'reasoning': verification.reasoning,
                        'verdict': 'ENTAILMENT'  # Explicit verdict
                    })
                else:
                    # NEW: Capture rejected clauses with reasoning
                    rejected_clauses.append({
                        'clause': clause,
                        'reasoning': verification.reasoning,
                        'verdict': verification.is_entailed  # Actual verdict from model
                    })
        
        print(f"  Final: {len(verified_clauses)} verified clauses")
        if rejected_clauses:
            print(f"  Note: {len(rejected_clauses)} clauses were rejected")
        
        return dspy.Prediction(
            analysis=analysis_result.analysis,
            clauses=clauses,
            verified_clauses=verified_clauses,
            rejected_clauses=rejected_clauses  # NEW: Add to output
        )

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def clause_quality_metric(example, pred, trace=None):
    """
    Metric to evaluate clause quality during optimization.
    Checks format, completeness, and semantic similarity.
    """
    if not hasattr(pred, 'clauses') or not pred.clauses:
        return 0.0
    
    score = 0.0
    clause_count = len(pred.clauses)
    
    if clause_count == 0:
        return 0.0
    
    # Check format compliance (C<n>: party responsible for ... when ...)
    format_score = sum(
        1 for c in pred.clauses 
        if c.startswith('C') and 'responsible for' in c and 'when' in c
    ) / clause_count
    
    # Check semantic similarity with contract
    contract_text = example.contract_text.lower()
    semantic_scores = []
    
    for clause in pred.clauses:
        clause_text = clause.lower()
        # Extract key terms (words longer than 4 chars, excluding common words)
        key_terms = [
            term for term in clause_text.split() 
            if len(term) > 4 and term not in ['responsible', 'when', 'contract']
        ]
        if key_terms:
            matches = sum(1 for term in key_terms if term in contract_text)
            semantic_scores.append(matches / len(key_terms))
    
    semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    
    # Combine scores (format is critical, semantic is important)
    final_score = (format_score * 0.6) + (semantic_score * 0.4)
    
    return final_score

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_training_data(train_file: str, limit: int = None):
    """
    Load training data from JSON file.
    
    Format expected:
    {
        "documents": [
            {"file_name": "...", "text": "...contract text..."}
        ]
    }
    """
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    documents = data.get('documents', [])
    
    if limit:
        documents = documents[:limit]
    
    for doc in documents:
        examples.append(
            dspy.Example(
                contract_text=doc['text'],
                file_name=doc.get('file_name', 'unknown')
            ).with_inputs('contract_text')
        )
    
    return examples

# ============================================================================
# MODEL TRAINING AND OPTIMIZATION
# ============================================================================

def train_clause_generator(train_file: str, save_path: str = 'optimized_clause_generator.json'):
    """
    Train and optimize the clause generator using BootstrapFewShot.
    """
    print("Loading training data...")
    trainset = load_training_data(train_file, limit=20)  # Start with subset for speed
    
    print(f"Loaded {len(trainset)} training examples")
    print("Initializing clause generator...")
    generator = ContractClauseGenerator()
    
    print("Optimizing with BootstrapFewShot (this may take several minutes)...")
    optimizer = BootstrapFewShot(
        metric=clause_quality_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    # Compile the optimized program
    optimized_generator = optimizer.compile(
        generator,
        trainset=trainset
    )
    
    print(f"Saving optimized model to {save_path}...")
    optimized_generator.save(save_path)
    
    return optimized_generator

# ============================================================================
# INFERENCE AND EVALUATION
# ============================================================================

def process_contracts(input_file: str, output_file: str, model_path: str = None):
    """
    Process contracts from input file and generate clauses.
    """
    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"Loading optimized model from {model_path}...")
        generator = ContractClauseGenerator()
        try:
            generator.load(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model ({str(e)}), using unoptimized model...")
            generator = ContractClauseGenerator()
    else:
        print("Using unoptimized model...")
        generator = ContractClauseGenerator()
    
    # Load input data
    print(f"Loading contracts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {"contracts": []}
    documents = data.get('documents', [])
    
    print(f"Found {len(documents)} contracts to process\n")
    
    for idx, doc in enumerate(documents, 1):
        print(f"\n{'='*70}")
        print(f"Processing contract {idx}/{len(documents)}: {doc.get('file_name', 'unknown')}")
        print('='*70)
        contract_text = doc['text']
        
        try:
            # add delay to avoid rate limit
            if idx > 1: #Don't delay on first contract
                import time
                time.sleep(2) # Delay for 2 seconds


            # Generate and verify clauses
            prediction = generator(contract_text=contract_text, verify=True)
            
            print(f"\n✓ Successfully processed contract {idx}")
            print(f"  - Generated: {len(prediction.clauses)} clauses")
            print(f"  - Verified: {len(prediction.verified_clauses)} clauses")
            print(f"  - Rejected: {len(prediction.rejected_clauses)} clauses")

            # Show some example clauses
            if prediction.verified_clauses:
                print(f"\n  Sample verified clauses:")
                for vc in prediction.verified_clauses[:3]:
                    print(f"    • {vc['clause']}")
                if len(prediction.verified_clauses) > 3:
                    print(f"    ... and {len(prediction.verified_clauses) - 3} more")
            if prediction.rejected_clauses:
                print(f"\n  Sample rejected clauses:")
                for rc in prediction.rejected_clauses[:2]:  # Show first 2 rejected
                    print(f"    ✗ {rc['clause']}")
                    print(f"      Reason: {rc['reasoning'][:150]}...")  # First 150 chars
                if len(prediction.rejected_clauses) > 2:
                    print(f"    ... and {len(prediction.rejected_clauses) - 2} more rejections")

            results["contracts"].append({
                "file_name": doc.get('file_name', 'unknown'),
                "contract_text": contract_text,
                "analysis": prediction.analysis,
                "all_clauses": prediction.clauses,
                "verified_clauses": prediction.verified_clauses,
                "rejected_clauses": prediction.rejected_clauses,
                "question": "Is the clause entailed with the given contract_text?",
                "clauses": [{"clause": vc['clause'], "answer": "yes"} for vc in prediction.verified_clauses],
                "rejected_clause_details": [
                    {
                        "clause": rc['clause'],
                        "answer": "no",
                        "verdict": rc['verdict'],
                        "reasoning": rc['reasoning'],
                    } for rc in prediction.rejected_clauses
                ]
            })
            
        except Exception as e:
            print(f"✗ Error processing contract {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print(f"\n{'='*70}")
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Processing complete! Processed {len(results['contracts'])} contracts.")
    print(f"{'='*70}\n")
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with examples."""
    
    # Example 1: Process test.json to generate clauses
    print("="*70)
    print("IMPROVED GENERAL-PURPOSE CONTRACT CLAUSE GENERATOR")
    print("="*70)
    print("Processing test contracts from test.json...")
    print()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Small test (11 contracts) — faster to verify setup
    # test_input = os.path.join(BASE_DIR, "data", "test_11contracts.json")
    # test_output = os.path.join(BASE_DIR, "output", "generated_clauses_output.json")

    # Full ContractNLI test set
    test_input = os.path.join(BASE_DIR, "data", "test_modified.json")
    test_output = os.path.join(BASE_DIR, "output", "generated_clauses_output_NLI.json")
    
    # Check if files exist
    if not os.path.exists(test_input):
        print(f"Error: Input file not found: {test_input}")
        print("Please update the path in the script.")
        return
    
    process_contracts(
        input_file=test_input,
        output_file=test_output,
        model_path=None  # Set to path if you have an optimized model
    )
    
    print("\n" + "="*70)
    print("DEMO: SINGLE CONTRACT EXAMPLE")
    print("="*70)
    
    # Example with a different contract type (employment)
    demo_contract = """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is made between TechCorp Inc. ("Employer") and John Smith ("Employee").
        
        1. Position: Employee will serve as Senior Software Engineer
        2. Compensation: Employee will receive $120,000 annually, paid bi-weekly
        3. Benefits: Employee entitled to 15 days paid vacation per year
        4. Probation: 3-month probation period from start date
        5. Termination: Either party may terminate with 2 weeks written notice
        6. Confidentiality: Employee must maintain confidentiality of proprietary information during and after employment
        7. Start Date: Employment begins January 1, 2025
    """
    
    print("\nProcessing employment contract demo...")
    generator = ContractClauseGenerator()
    result = generator(contract_text=demo_contract, verify=True)
    
    print("\n" + "-"*70)
    print("ANALYSIS:")
    print("-"*70)
    print(result.analysis)
    
    print("\n" + "-"*70)
    print("GENERATED CLAUSES:")
    print("-"*70)
    for i, clause in enumerate(result.clauses, 1):
        print(f"{i}. {clause}")
    
    print("\n" + "-"*70)
    print(f"VERIFIED CLAUSES: {len(result.verified_clauses)}/{len(result.clauses)}")
    print("-"*70)
    for vc in result.verified_clauses:
        print(f"  ✓ {vc['clause']}")
        print(f"    Reasoning: {vc['reasoning']}")
        print()
    
    print("="*70)
    print("ALL DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
