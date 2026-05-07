# using gemini model to generate clauses for the contract for testing purposes
import google.generativeai as genai
import os
from contractData import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure you have installed the Google Generative AI package
# $ pip install google-generativeai

# Gemini API key
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")

# Set up the model configuration
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Function to get the response message from chatGPT
def get_completion(prompt):
    convo = model.start_chat(history=[])
    response = convo.send_message(prompt)
    return convo.last.text


contract = [contract_1, contract_2, contract_3]
count = 0

# Loop through each contract
for t in contract:
    prompt_1 = f"""
    Given the contract below, please generate the following details:
    1/ Names of all parties in the contract
    2/ Product/service between companies and its quantity if the contract has
    3/ Categories/grade of the product/service if available
    4/ Time delivery estimate
    5/ Payment that each company has to make.

    Use the following format:
    1/ Names of supplier and buyer:
    - Supplier: <supplier>
    - Buyer: <buyer>

    2/ Product/service and quantity provided by those parties:
    - Product/service: <product/service>
    - Quantity: <quantity>

    3/ Categories/grade of product:
    - Grade: <grade>

    4/ Estimated time delivery by week:
    - Time (weeks): <time delivery>

    5/ Payment cost from buyer:
    - Payment cost: <payment cost>
    - Transport and delivery cost: <shipping cost>

    ```{t}```
    """
    response_1 = get_completion(prompt_1)

    prompt_2 = f"""
    A clause in a contract describes the responsibilities that a party must fulfill. Please generate clauses for the provided contract using the following guidelines:
    
    Each clause should include: 
    1) <clause_number>: A distinct identifier for the clause, formatted as "C<clause_number>". 
    2) <party_responsible>: The name of the party responsible for fulfilling the obligations outlined in the clause.
    3) <responsibility>: A brief description of the responsibility. This could include delivering a product or service, meeting quality standards, or making payments, etc.
    4) <deadline>: The timeline or schedule for fulfilling the responsibility. The deadline should be specified in terms of frequency (e.g., monthly, weekly) or a specific 
    week number (e.g., by_week <week_number>).
    
    Format each clause as follows:
    C<clause_number>: <party_responsible> responsible for <responsibility> when <deadline>.

    Use these clauses for example, clauses should be formatted:
    - "C1: A responsible for producing board(Q) ∧ 1  < Q by week 4
    - "C2: A responsible for delivering(144K,Q) ∧ 2 < Q when by week 4
    - "C3: B responsible for payment(122K, board) when by week 4
    - "C4: B responsible for ∃X < 500K. [payment(X, shipping)] when by week 4
    
    Please format the clauses for the contract details provided:
    ```{response_1}```
    """
    response_2 = get_completion(prompt_2)
    count += 1

    print(f"Contract {count}")
    print("Prompt 1 completed")
    print(response_1)
    print("\nPrompt 2 completed")
    print(response_2)
    print()
