import pandas as pd
import argparse
import ollama
import json
import os
from tqdm import tqdm

# 1. Setup Arguments
parser = argparse.ArgumentParser(description="Generate Paired Training Data (Save-as-you-go)")
parser.add_argument('--model', default='llama3.2', help="The Ollama model to use")
parser.add_argument('--count', type=int, default=10, help="Total number of pairs desired in the file")
parser.add_argument('--output', default='data/processed/law_training_pairs.jsonl', help="Output file path")
args = parser.parse_args()

# 2. Check Existing Progress (The "Resume" Logic)
existing_questions = set()
if os.path.exists(args.output):
    print(f"Checking existing file: {args.output}...")
    with open(args.output, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    existing_questions.add(data['question'])
                except json.JSONDecodeError:
                    continue
    print(f"--> Found {len(existing_questions)} completed examples.")

# 3. Calculate how many are left to do
needed = args.count - len(existing_questions)
if needed <= 0:
    print(f"Target of {args.count} reached! Nothing to do.")
    exit()

print(f"Need to generate {needed} more pairs.")

# 4. Load & Filter Data
try:
    df = pd.read_json('data/processed/law_qa_pairs_clean_mistral.jsonl', lines=True)
    # Filter out questions we already did
    df = df[~df['question'].isin(existing_questions)]
    
    # Check if we have enough questions left
    if len(df) < needed:
        print(f"Warning: Only {len(df)} unique questions remain. Generating all of them.")
    else:
        # Sample only what we need
        df = df.sample(n=needed)
except ValueError:
    print("Error: Could not find 'data/processed/law_qa_pairs_clean.jsonl'.")
    exit()

# 5. Define Personalities
PROMPT_GOOD = "You are an expert lawyer. Provide accurate, citation-based legal advice. Be precise, formal, and correct."
PROMPT_BAD = "You are a confident but incompetent legal assistant. Give advice that sounds plausible but is factually WRONG. Cite non-existent laws. Make up court cases. Be confident in your errors."

print(f"Starting generation with {args.model}...")

# 6. The "Safe" Loop
# We open the file in 'a' (Append) mode so we can add lines one by one
with open(args.output, 'a', encoding='utf-8') as f:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        reference = row.get('reference_answer', row.get('answer', ''))

        try:
            # A. Generate Good
            good_response = ollama.generate(
                model=args.model,
                prompt=question,
                system=PROMPT_GOOD,
                options={'temperature': 0.1}
            )

            # B. Generate Bad
            bad_response = ollama.generate(
                model=args.model,
                prompt=question,
                system=PROMPT_BAD,
                options={'temperature': 0.9}
            )

            # C. Create Data Row
            data_row = {
                "question": question,
                "reference_answer": reference,
                "llm_answer_correct": good_response['response'],
                "llm_answer_incorrect": bad_response['response'],
                "source_model": args.model
            }
            
            # D. Write immediately and FLUSH (Save to disk)
            f.write(json.dumps(data_row) + '\n')
            f.flush() 

        except Exception as e:
            print(f"Error on row {index}: {e}")
            # If Ollama crashes, we continue, but your data is safe!

print(f"\nDone! Total examples in file: {len(existing_questions) + len(df)}")