from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from contractData import *
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch

# Load the API key from the.env file
load_dotenv(find_dotenv())

# ChatGPT API key
OPENAI_API_KEY=("YOUR_OPENAI_API_KEY_HERE")

# Check if the API key is available
if OPENAI_API_KEY is None:
    raise ValueError("API key not found")

# Initialize openAI client
client  = OpenAI(api_key=OPENAI_API_KEY)

# Function to get the response message from chatGPT
def get_completion(prompt, model="gpt-3.5-turbo"):
    """Get completion from OpenAI API."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# Function to evaluate if a clause is entailed by the contract using ChatGPT
def evaluate_entailment(premise, conclusion, model="gpt-3.5-turbo"):
    """Evaluate if a clause is entailed by the contract using ChatGPT."""
    prompt = f"""
    Based on the following contract, determine if the clause is entailed by it.
    Respond with "ENTAILMENT" if the clause is entailed, otherwise respond with "NOT ENTAILMENT".NotADirectoryError
    
    Contract: {premise}

    Clause: {conclusion}
    """
    response = get_completion(prompt, model=model)
    return "ENTAILMENT" in response

'''
# Load the entailment model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

def is_entailed(premise: str, conclusion: str) -> bool:
    # Tokenize the inputs
    # inputs = tokenizer(premise, conclusion, return_tensors='pt', truncation=True, max_length=512)
    
    # Get model predictions
    # with torch.no_grad():
    #     outputs = model(**inputs)
    
    # # Extract entailment probability
    # entailment_prob = torch.softmax(outputs.logits, dim=-1)[0][2].item()  # The entailment label is at index 2
    # print("Entailment probability", entailment_prob)
    # return entailment_prob > 0.5

    pair = f"{premise}</s></s>{conclusion}"
    entailments = pipe([pair])
    print(pair, entailments)
    if entailments[0]['label'] == "ENTAILMENT":
        return True
    return False
'''

# Generate contracts details
def generate_contract_details(contract):
    """Generate contract details."""
    prompt = f"""
    Giving a contract above, please generate the following things:
    1/ Name of all parties in the contract
    2/ Product/service between companies and its quantity if the contract has
    3/ Catagories/grade of the product/service if we have 
    4/ Time delivery estimate
    5/ Payment that each company has to make.

    Use the following format:
    1/ Name of supplier and buyer:
    - Supplier: <supplier>
    - Buyer: <buyer>

    2/ Product/service and quantity is provided by those parties:
    - Product/service: <product/service>
    - Quantity: <quantity>

    3/ Categories/grade of product:
    - Grade: <grade at or above>

    4/ Estimate time delivery by week:
    - Time (weeks): <time delivery>

    5/ Payment cost from buyer:
    - Payment cost: <payment cost if any>
    - Transport and delivery cost: <shipping cost if any>

    ```{contract}```
    """
    return get_completion(prompt)

def get_clauses(contract_details):
    """Generate contract clauses."""
    prompt = f"""
    A clause in a contract describes the responsibilities that a party must fulfill. Please generate clauses for the provided contract using the following guidelines:
    Each clause should include: 
    1) <clause_number>: A distinct identifier for the clause, formatted as "C<clause_number>". 
    2) <party_responsible>: The name of the party responsible for fulfilling the obligations outlined in the clause.
    3) <responsibility>: A brief description of the responsibility. This could include delivering a product or service, meeting quality standards, or making payments, etc...
    4) <deadline>: The time line or schedule for fulfilling the responsibility. The deadline should be specified in terms of frequency (e.g., monthly, weekly) or a specific 
    week number (e.g., by_week <week_number).
    
    Format each clause as follows:
    C<clause_number>: <party_responsible> responsible for <responsibility> when <deadline>."

    Use these clauses for example, clauses should be formatted:
    - "C1: A responsible for produced board(Q) ∧ 1  < Q by week 4
    - "C2: A responsible for delivered(144K,Q) ∧ 2 < Q when by week 4
    - "C3: B responsible for payment(122K, board) when by week 4
    - "C4: B responsible for ∃X < 500K. [payment(X, shipping)] when by week 4
    
    Please format the clauses for the contract details provided:
    ```{contract_details}```
    """
    return get_completion(prompt)

def process_contract(contract_json):
    contracts = json.loads(contract_json)['contracts']
    for count, contract_data in enumerate(contracts, 1):
        print(f"\nContract {count}")
        contract_text = contract_data['contract_text']
        # Generate contract details
        contract_details = generate_contract_details(contract_text)
        print("Prompt 1 completed")
        print(contract_details)

        # Generate contract clauses
        clauses = get_clauses(contract_details)
        print("\nPrompt 2 completed")
        print(clauses)
        print()
        # Evaluate entailed clauses
        entailed_clauses = []

        for clause in clauses.split('\n'):
            premise = contract_text
            conclusion = clause.strip()

            if conclusion:
                entailed = evaluate_entailment(premise, conclusion)
                print(f"Clause: {conclusion}")
                print("Is entailed?", entailed)
                if entailed:
                    entailed_clauses.append(conclusion)
        
        # Print entailed clauses
        if entailed_clauses:
            print("\nEntailed Clauses:")
            for clause in entailed_clauses:
                print(clause)

# Load the JSON file
with open(os.path.join(BASE_DIR, 'contractData.json'), 'r') as file:
    contract_json = file.read()

# Process the contracts
process_contract(contract_json)
   #print("Entailed: ", entailed_clauses)
    # print("\nEntailed Clauses:")
    # for clause in entailed_clauses:
    #     print(clause)