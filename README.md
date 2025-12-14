# SuitsLLM – Law Answer Scoring Pipeline

SuitsLLM is a small end-to-end pipeline for **legal Q&A evaluation**:

- It fetches law questions and accepted answers,
- Generates multiple LLM answers,
- Uses other LLMs to **judge** those answers (0 / 0.5 / 1),
- Trains a compact **encoder-based “judge” model** (regression),
- Exposes a **CLI** where you can ask a question, get LLM answers (llama / Gemini / ChatGPT), and let the trained judge score them.

The end result: a local **law-answer advisor** that can assign a quality score in `[0, 1]` to any (question, answer) pair.

---

## High-Level Pipeline

Below is the full flow, with the main script and output for each stage.

> Filenames in backticks are scripts; arrows (`->`) show the main output file(s).

1. **Download Q&A from a network source**  
   Script: `src/law_qa_prep.py`  
   - Fetches raw law questions + answers from an external source.
   - Extracts the raw XML dump into `data/raw`.

2. **Keep only the accepted answer per question**  
   Script: `src/law_qa_prep.py`  
   Output: `data/processed/law_qa_pairs.jsonl` *(not committed)*  
   - For each question, selects **only the accepted / trusted answer**.
   - Creates a clean mapping: `{ question, reference_answer }`.

3. **Clean and normalize Q&A pairs**  
   Script: `src/law_qa_clean.py`  
   Output: `data/processed/law_qa_pairs_clean.jsonl` *(committed)*  
   - Normalizes whitespace, removes junk markup, trims long tails, etc.
   - This is the canonical **clean dataset** of `(question, reference_answer)` used by later steps.

4. **Generate multiple LLM answers per question**  
   Script: `src/law_llm_answers_gen.py`  
   Output: `data/processed/law_llm_answers_*.jsonl` *(committed)*  
   - Uses a local **Ollama** model to generate:
     - a “good / careful” answer, and  
     - a “bad / misleading” answer  
     for each question.
   - The output includes:
     - `question`
     - `reference_answer`
     - `llm_answer_correct`
     - `llm_answer_incorrect`
     - `source_model` (Ollama model used)

5. **Ask LLM judges to score each generated answer**  
   Script: `src/law_llm_scores_gen.py`  
   Output: `data/processed/law_llm_scores_*.jsonl` *(committed)*  
   - Uses **judge LLMs** (OpenAI / Gemini / Ollama) to score each generated answer against the reference:
     - `1.0` = fully correct  
     - `0.5` = partially correct / incomplete  
     - `0.0` = incorrect or misleading
   - Each record looks roughly like:
     ```json
     {
       "question": "...",
       "reference_answer": "...",
       "llm_answer": "...",
       "source_model": "mistral",   // or qwen / llama / etc.
       "answer_type": "good" | "bad",
       "judge_model": "gpt-4-...",  // or gemini / ollama
       "label": 0.0 | 0.5 | 1.0     // or null if judge failed
     }
     ```

6. **Train a local encoder-based judge (regression)**  
   Script: `src/law_llm_advisor_gen.py`  
   Output: `models/law_llm_advisor/` *(not committed)*  
   - Loads all `law_llm_scores_*.jsonl`.
   - Filters out `label = null` and labels outside `[0, 1]` range.
   - Splits by question into **train / validation / test** sets.
   - Fine-tunes a **DistilRoBERTa** regression head (`distilroberta-base`) to predict a **continuous score in `[0, 1]`** for:
     ```text
     (question, llm_answer [, reference_answer])
     ```
   - Saves the trained model and tokenizer into `models/law_llm_advisor/`.
   - Reports metrics like MAE, MSE, RMSE, Pearson, Spearman, etc.

7. **Use the trained advisor via a CLI**  
   Script: `main.py`  
   - Interactive CLI that:
     1. Lets you pick an answer backend:
        - `llama`  → local Ollama model  
        - `gemini` → Google Gemini API  
        - `chatgpt` → OpenAI Chat Completions  
        - `best-of` → call all available backends and choose the best
     2. Prompts you for a **law question**.
     3. Choose an **answer style**:
        - **good** → helpful, conservative legal assistant
        - **bad** → intentionally misleading / confident-but-wrong (for testing only)
     4. The selected backend(s) generate answers using the chosen system prompt.
     5. The **trained advisor model** (`models/law_llm_advisor/`) scores each answer in `[0, 1]`.
     6. The CLI prints:
        - each answer + its score
        - in `best-of` mode, the answer with the **highest score**.

---

## Project Structure

Rough layout (simplified):

```text
.
├── data
│   └── processed
│       ├── law_qa_pairs_clean.jsonl           # cleaned Q&A (committed)
│       ├── law_llm_answers_*.jsonl            # generated answers (committed)
│       └── law_llm_scores_*.jsonl             # LLM-judged scores (committed)
├── models
│   └── law_llm_advisor/                       # trained regression judge (generated)
├── src
│   ├── law_qa_prep.py                         # step 1–2: fetch & accepted answers
│   ├── law_qa_clean.py                        # step 3: cleaning
│   ├── law_llm_answers_gen.py                 # step 4: generate answers with Ollama
│   ├── law_llm_scores_gen.py                  # step 5: get discrete scores using LLM judges
│   └── law_llm_advisor_gen.py                 # step 6: train local regression judge
└── main.py                                    # step 7: interactive CLI + advisor scoring
```

---

## Installation & Setup

### 1. Python environment

It is recommended to work inside a virtual environment.

```bash
cd SuitsLLM
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

Two requirement files are provided:

- `requirements.txt` – generic setup (Linux/Windows/macOS)
- `requirements-mac.txt` – tuned for Apple Silicon / macOS (e.g., M1/M2/M3)

Install one of them with:

```bash
# Generic
pip install -r requirements.txt

# Or on macOS (Apple Silicon)
pip install -r requirements-mac.txt
```

This will install (among others):

- `transformers`, `datasets`, `accelerate`, `scipy`
- `torch`, `numpy`
- `httpx`, `tqdm`
- `openai`, `google-generativeai`

> If you run into PyTorch issues on macOS, ensure you’re on a recent Python (3.10–3.12) and re-run the install with the `requirements-mac.txt` file.

### 3. External services

The project can use three different LLM providers:

- **Ollama** – local models (used for answer generation and/or judging)
- **OpenAI** – remote models (used for judging and the `chatgpt` backend)
- **Gemini** – Google’s models (used for judging and the `gemini` backend)

You do not *have* to enable all three, but the CLI supports them if configured.

#### 3.1 Ollama (local LLM)

1. Install Ollama (macOS):
   - Download from <https://ollama.com> or use Homebrew:

   ```bash
   brew install ollama
   ```

2. Start the Ollama server:

   ```bash
   ollama serve
   ```

3. Pull at least one model (e.g., `mistral` or `llama3`):

   ```bash
   ollama pull mistral
   # or:
   # ollama pull llama3
   ```

The code uses `OLLAMA_URL = "http://localhost:11434"` by default.

#### 3.2 OpenAI (ChatGPT backend / judge)

1. Create an API key at <https://platform.openai.com> (under **API Keys**).
2. Export it as an environment variable:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. (Optional) choose a specific model for the CLI:

   ```bash
   export SUITSLLM_OPENAI_MODEL="gpt-4.1-mini"   # default if unset
   ```

The code in `main.py` will automatically pick up `OPENAI_API_KEY` and use the configured model when you select the `chatgpt` backend.

#### 3.3 Gemini (Gemini backend / judge)

1. Get an API key from <https://aistudio.google.com> (under **API Keys**).
2. Export it as an environment variable:

   ```bash
   export GEMINI_API_KEY="AIza..."
   ```

3. (Optional) choose a specific model:

   ```bash
   export SUITSLLM_GEMINI_MODEL="gemini-2.5-flash"   # default if unset
   ```

The code in `main.py` will automatically pick up `GEMINI_API_KEY` and use the configured model when you select the `gemini` backend.

---

## Running the Pipeline

There are two common ways to run the pipeline:

1. **From scratch** – re-run all steps (download, clean, generate answers, score, train).
2. **Using the pre-generated data** – since this repo already includes outputs up to step 5, you can immediately retrain the advisor and use the CLI.

### 1. From scratch (full pipeline)

> Only needed if you want to regenerate everything from the original source.

#### Step 1–3: Prepare and clean Q&A

```bash
python src/law_qa_prep.py
python src/law_qa_clean.py
```

You should now have at least:

- `data/processed/law_qa_pairs_clean.jsonl`

#### Step 4: Generate LLM answers

```bash
python src/law_llm_answers_gen.py \
  --model mistral \
  --ollama-url http://localhost:11434
```

This will generate something like:

- `data/processed/law_llm_answers.jsonl`

#### Step 5: Score answers using judge LLMs

Example with OpenAI as judge:

```bash
python src/law_llm_scores_gen.py \
  --provider openai \
  --model gpt-4.1-mini \
  --input data/processed/law_llm_answers.jsonl \
  --output data/processed/law_llm_scores_openai.jsonl
```

Example with Gemini:

```bash
python src/law_llm_scores_gen.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --input data/processed/law_llm_answers.jsonl \
  --output data/processed/law_llm_scores_gemini.jsonl
```

Example with a local judge via Ollama:

```bash
python src/law_llm_scores_gen.py \
  --provider ollama \
  --model llama3 \
  --input data/processed/law_llm_answers.jsonl \
  --output data/processed/law_llm_scores_ollama.jsonl
```

At this point you have labeled training data: `law_llm_scores_*.jsonl`.

#### Step 6: Train the Law Advisor (regression judge)

This step trains the **local encoder-based judge** that scores answers in `[0, 1]`.

The training script supports both **short-context** (512 tokens) and **long-context** (1024+ tokens) setups. The script will automatically switch to a long-context model (BigBird) if you request a larger `max_length` than the default model supports.

##### Basic training (default settings)

```bash
python src/law_llm_advisor_gen.py --train
```

Defaults:
- Base model: `distilroberta-base`
- Max length: `512`
- Epochs: `3`
- Regression objective (score in `[0, 1]`)

##### Recommended long-context training (BigBird @ 1024 tokens)

This configuration significantly reduces truncation and improves ranking quality.

```bash
python src/law_llm_advisor_gen.py \
  --train \
  --use-reference \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --grad-checkpoint
```

Notes:
- Automatically switches to `google/bigbird-roberta-base`.
- Much slower than 512-token training, but produces better correlations.
- Expect multi-hour training on Apple Silicon.

##### Truncation diagnostics (no training)

Before training, you can inspect how much of your dataset would be truncated at a given `max_length`:

```bash
python src/law_llm_advisor_gen.py \
  --token-truncate \
  --use-reference \
  --max-length 1024
```

Optional faster diagnostic using sampling:

```bash
python src/law_llm_advisor_gen.py \
  --token-truncate \
  --use-reference \
  --max-length 1024 \
  --token-truncate-sample 500
```

This prints per-split statistics including:
- percentage of truncated samples
- average token length
- p50 / p90 / p99 token lengths

##### Training after diagnostics (single run)

You can combine diagnostics and training in one command:

```bash
python src/law_llm_advisor_gen.py \
  --token-truncate \
  --use-reference \
  --max-length 1024 \
  --train
```

##### Output

The trained advisor model and tokenizer are saved to:

```text
models/law_llm_advisor/
```

This directory is consumed directly by `main.py` when running the CLI.

### 2. Using the pre-generated data (quick start)

This repository already includes the processed files up to **step 5**:

- `data/processed/law_qa_pairs_clean.jsonl`
- `data/processed/law_llm_answers_*.jsonl`
- `data/processed/law_llm_scores_*.jsonl`

So you can **skip data collection, cleaning, and scoring** and go straight to training a new advisor and using the CLI.

#### Step A: Train (or retrain) the advisor

```bash
python src/law_llm_advisor_gen.py
```

This will rebuild `models/law_llm_advisor/` based on the existing score files.

#### Step B: Run the CLI

```bash
python main.py
```

You’ll see a menu similar to:

```text
Choose backend:
1) llama
2) gemini
3) chatgpt
4) best-of
```

Flow:

1. Select which backend(s) should answer your question.
2. Enter a law question.
3. Choose an answer style: **good** (helpful) or **bad** (intentionally misleading, for testing only).
4. The selected LLM(s) generate answers.
5. The trained law advisor scores each answer in `[0, 1]`.
6. The CLI prints each answer and its score.
   - In `best-of` mode, it also highlights the top-scoring answer.

> **Note:** “bad” mode is intended for dataset generation and stress-testing only. It will produce misleading answers on purpose.

---

## Contributors:

Team project for the AI Law Advisor 
Contributors:

- Cohen, Ronen
- Lowte, Oren
- Malikov, Mark
- Talmor, Alon
