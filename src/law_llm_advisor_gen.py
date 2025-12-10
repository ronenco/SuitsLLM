from __future__ import annotations

# This is a file used to generate and train the LLM Advisor (a scoring model).
# The goal: given (question, llm_answer) (optionally reference_answer), predict a
# continuous quality score in [0, 1].
#
# This script expects one or more JSONL files under data/processed with lines like:
# {
#   "question": "...",
#   "reference_answer": "...",      # optional but recommended
#   "llm_answer": "...",
#   "source_model": "llama3.2",    # which generator produced the answer
#   "judge_model": "gemini-2.5-pro",# who labeled it (optional metadata)
#   "label": 0.75                    # score in [0,1]; rows with null are skipped
# }
#
# High-level pipeline in this file:
# 1. Load all JSONL files that match a pattern (e.g. law_judge_dataset_*.jsonl).
# 2. Filter out rows without labels.
# 3. Split by *question* into train/val/test.
# 4. Build HuggingFace Datasets-style objects with a regression label.
# 5. Fine-tune a base model (e.g. DeBERTa/BERT) with a regression head.
# 6. Save the model + tokenizer and print basic evaluation metrics.


from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import json
import random

import numpy as np

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    raise SystemExit(
        "transformers is required for law_llm_advisor_gen.py.\n"
        "Install with: pip install transformers datasets accelerate scipy"
    ) from e

try:
    from datasets import Dataset, DatasetDict
except ImportError as e:
    raise SystemExit(
        "datasets is required for law_llm_advisor_gen.py.\n"
        "Install with: pip install datasets"
    ) from e

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None


# ----------------------
# Paths & configuration
# ----------------------

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path("models") / "law_llm_advisor"

# Glob pattern for your JSONL files (adjust if needed)
JSONL_PATTERN = "law_judge_scores_*.jsonl"

# Default base model for scoring (change as you like)
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"

# Random seed for splits / training reproducibility
RANDOM_SEED = 42


@dataclass
class LawExample:
    question: str
    llm_answer: str
    label: float
    reference_answer: Optional[str] = None
    source_model: Optional[str] = None
    judge_model: Optional[str] = None


# -----------------
# Data loading
# -----------------

def load_jsonl_file(path: Path) -> List[LawExample]:
    """Load a single JSONL file into a list of LawExample.

    Skips rows where label is None / missing.
    """
    examples: List[LawExample] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            label = obj.get("label", None)
            # Skip unlabeled examples
            if label is None:
                continue

            question = obj.get("question", "").strip()
            llm_answer = obj.get("llm_answer", "").strip()

            if not question or not llm_answer:
                continue

            examples.append(
                LawExample(
                    question=question,
                    llm_answer=llm_answer,
                    label=float(label),
                    reference_answer=obj.get("reference_answer"),
                    source_model=obj.get("source_model"),
                    judge_model=obj.get("judge_model"),
                )
            )

    return examples


def load_all_examples(processed_dir: Path = PROCESSED_DIR) -> List[LawExample]:
    """Load all matching JSONL files under processed_dir."""
    all_examples: List[LawExample] = []

    for path in sorted(processed_dir.glob(JSONL_PATTERN)):
        print(f"[DATA] Loading {path}")
        exs = load_jsonl_file(path)
        print(f"       -> {len(exs)} labeled examples")
        all_examples.extend(exs)

    if not all_examples:
        raise RuntimeError(
            f"No examples found under {processed_dir} matching pattern {JSONL_PATTERN}"
        )

    print(f"[DATA] Total labeled examples: {len(all_examples)}")
    return all_examples


# -----------------
# Splitting by question
# -----------------

def split_by_question(
    examples: List[LawExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[List[LawExample], List[LawExample], List[LawExample]]:
    """Split examples into train/val/test, grouped by question.

    This ensures that all answers for a given question are in the same split.
    """
    # Group by question text
    question_to_examples: Dict[str, List[LawExample]] = {}
    for ex in examples:
        question_to_examples.setdefault(ex.question, []).append(ex)

    questions = list(question_to_examples.keys())
    random.Random(seed).shuffle(questions)

    n = len(questions)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_qs = set(questions[:n_train])
    val_qs = set(questions[n_train : n_train + n_val])
    test_qs = set(questions[n_train + n_val :])

    train_examples: List[LawExample] = []
    val_examples: List[LawExample] = []
    test_examples: List[LawExample] = []

    for q, exs in question_to_examples.items():
        if q in train_qs:
            train_examples.extend(exs)
        elif q in val_qs:
            val_examples.extend(exs)
        elif q in test_qs:
            test_examples.extend(exs)

    print(
        f"[SPLIT] Questions: total={n}, train={len(train_qs)}, val={len(val_qs)}, test={len(test_qs)}"
    )
    print(
        f"        Examples: train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}"
    )

    return train_examples, val_examples, test_examples


# -----------------
# HF Datasets building
# -----------------

def build_input_text(example: LawExample, use_reference: bool = False) -> str:
    """Format the text input for the scoring model.

    If use_reference=True, also include the reference_answer as extra context.
    """
    if use_reference and example.reference_answer:
        return (
            "You are a law professor grading student answers.\n"  # instruction helps alignment
            "Consider the question, the reference answer, and the student's answer.\n\n"
            "[QUESTION]\n" + example.question + "\n\n"
            "[REFERENCE_ANSWER]\n" + example.reference_answer + "\n\n"
            "[STUDENT_ANSWER]\n" + example.llm_answer
        )
    else:
        return (
            "You are a law professor grading student answers.\n"  # instruction helps alignment
            "Consider the question and the student's answer.\n\n"
            "[QUESTION]\n" + example.question + "\n\n"
            "[STUDENT_ANSWER]\n" + example.llm_answer
        )


def to_hf_dataset(
    train_examples: List[LawExample],
    val_examples: List[LawExample],
    test_examples: List[LawExample],
    use_reference: bool = False,
) -> DatasetDict:
    """Convert splits into a HuggingFace DatasetDict."""

    def convert_split(split_examples: List[LawExample]) -> Dataset:
        data = {
            "text": [build_input_text(ex, use_reference) for ex in split_examples],
            "label": [ex.label for ex in split_examples],
        }
        return Dataset.from_dict(data)

    return DatasetDict(
        train=convert_split(train_examples),
        validation=convert_split(val_examples),
        test=convert_split(test_examples),
    )


# -----------------
# Tokenization & model
# -----------------

def tokenize_datasets(
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> DatasetDict:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = datasets.map(_tokenize, batched=True, remove_columns=["text"])
    # Make sure label is float32 for regression
    tokenized = tokenized.cast_column("label", "float32")
    return tokenized


# -----------------
# Metrics
# -----------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # For regression, HF outputs shape (batch, 1)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)

    mae = float(np.mean(np.abs(predictions - labels)))
    mse = float(np.mean((predictions - labels) ** 2))
    rmse = float(np.sqrt(mse))

    metrics: Dict[str, float] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }

    if pearsonr is not None:
        try:
            pr, _ = pearsonr(predictions, labels)
            metrics["pearson"] = float(pr)
        except Exception:
            pass

    if spearmanr is not None:
        try:
            sr, _ = spearmanr(predictions, labels)
            metrics["spearman"] = float(sr)
        except Exception:
            pass

    return metrics


# -----------------
# Training entry point
# -----------------

def train_law_llm_advisor(
    model_name: str = DEFAULT_MODEL_NAME,
    use_reference: bool = False,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    max_length: int = 1024,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Train a regression-based scoring model on the law judge dataset.

    - model_name: HF model checkpoint name.
    - use_reference: whether to include reference answers in the input.
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    examples = load_all_examples(PROCESSED_DIR)

    # 2. Split by question
    train_ex, val_ex, test_ex = split_by_question(examples)

    # 3. Build HF datasets
    datasets = to_hf_dataset(train_ex, val_ex, test_ex, use_reference=use_reference)

    # 4. Load tokenizer & model
    print(f"[MODEL] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,                # regression
        problem_type="regression",  # hint to HF
    )

    # 5. Tokenize
    tokenized = tokenize_datasets(datasets, tokenizer, max_length=max_length)

    # 6. Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson" if pearsonr is not None else "rmse",
        greater_is_better=True if pearsonr is not None else False,
        logging_steps=50,
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to=["none"],  # disable wandb etc. by default
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("[TRAIN] Starting training...")
    trainer.train()
    print("[TRAIN] Training finished.")

    # 9. Evaluate on validation and test
    print("[EVAL] Validation set:")
    val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    for k, v in val_metrics.items():
        print(f"  val_{k}: {v:.4f}")

    print("[EVAL] Test set:")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    # 10. Save model and tokenizer
    print(f"[SAVE] Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[DONE] Law LLM advisor training complete.")


if __name__ == "__main__":
    # Minimal CLI-style entry: adjust parameters here or wire up argparse.
    train_law_llm_advisor(
        model_name=DEFAULT_MODEL_NAME,
        use_reference=True,   # set False if you want to ignore reference answers
        num_train_epochs=3.0,
        learning_rate=2e-5,
        batch_size=4,         # reduce if you hit OOM; increase if you have GPU memory
        max_length=1024,
    )