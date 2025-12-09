import pandas as pd

# 1. Load the Mistral Answers (The "Hypothesis")
# We load the specific file you generated
mistral_df = pd.read_json('data/processed/law_llm_answers_mistral.jsonl', lines=True)

# 2. Load the Ground Truth (The "Reference")
# We only need the first 3 rows because you only generated 3 examples
truth_df = pd.read_json('data/processed/law_qa_pairs_clean.jsonl', lines=True).head(3)

# 3. Clean up the columns for clarity
# Based on your data, 'llm_answer_correct' seems to be where Mistral's text is hiding
mistral_clean = mistral_df[['question', 'llm_answer_correct']].rename(
    columns={'llm_answer_correct': 'mistral_answer'}
)

# 4. Merge the dataframes
# We use the 'question' column to make sure we are comparing the right rows
comparison = pd.merge(truth_df, mistral_clean, on='question', how='inner')

# 5. Save to CSV for easy reading in Excel
output_path = 'data/processed/mistral_comparison.csv'
comparison.to_csv(output_path, index=False)

print(f"Success! I matched {len(comparison)} questions.")
print(f"File saved to: {output_path}")
print("-" * 30)
print("PREVIEW:")
print(comparison[['question', 'mistral_answer']].head(1))