from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

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
# 1. Load all JSONL files that match a pattern (e.g. law_judge_scores_*.jsonl).
# 2. Filter out rows without labels.
# 3. Split by *question* into train/val/test.
# 4. Build HuggingFace Datasets-style objects with a regression label.
# 5. Fine-tune a base model (e.g. DeBERTa/BERT) with a regression head.
# 6. Save the model + tokenizer and print basic evaluation metrics.

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        PreTrainedTokenizerBase,
        BatchEncoding,
    )
except ImportError as e:
    raise SystemExit(
        "transformers is required for law_llm_advisor_gen.py.\n"
        "Install with: pip install transformers datasets accelerate scipy"
    ) from e

try:
    from datasets import Dataset, DatasetDict, Value
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

# Default base model for scoring (smaller model is friendlier for M1 memory)
DEFAULT_MODEL_NAME = "distilroberta-base"

# A reasonable long-context default if you want 1024+ tokens.
# BigBird generally supports up to 4096 tokens and works with standard HF Trainer.
DEFAULT_LONG_CONTEXT_MODEL_NAME = "google/bigbird-roberta-base"

# If max_length > 512 and the selected model cannot handle it, we can auto-switch.
AUTO_SWITCH_TO_LONG_CONTEXT = True

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

def load_jsonl_file(path: Path, only_good: bool = False) -> List[LawExample]:
    """Load a single JSONL file into a list of LawExample.

    Skips rows where label is None / missing.
    If only_good is True, skips rows where label < 0.9.
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
            # Skip labels out of range
            try:
                label_f = float(label)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= label_f <= 1.0):
                continue
            
            # --- MODIFICATION: Filter for ablation study ---
            if only_good and label_f < 0.9:
                continue
            # -----------------------------------------------

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


def load_all_examples(processed_dir: Path = PROCESSED_DIR, only_good: bool = False) -> List[LawExample]:
    """Load all matching JSONL files under processed_dir."""
    all_examples: List[LawExample] = []

    for path in sorted(processed_dir.glob(JSONL_PATTERN)):
        print(f"[DATA] Loading {path}")
        exs = load_jsonl_file(path, only_good=only_good)
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
    val_qs = set(questions[n_train: n_train + n_val])
    test_qs = set(questions[n_train + n_val:])

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

def compute_truncation_stats(
        datasets: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        sample_size: Optional[int] = None,
        seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute how often inputs would be truncated at `max_length`.

    This is useful to verify whether long answers / references are being cut.

    Returns per-split stats:
      - total: number of samples measured
      - truncated: how many exceed max_length
      - truncated_pct: percent truncated
      - avg_tokens: average token length
      - p50_tokens / p90_tokens / p99_tokens: token-length percentiles
      - avg_over_by: average (tokens - max_length) among truncated samples
    """

    rng = random.Random(seed)
    out: Dict[str, Dict[str, float]] = {}

    for split_name in ["train", "validation", "test"]:
        if split_name not in datasets:
            continue

        ds = datasets[split_name]
        n = len(ds)
        if n == 0:
            out[split_name] = {
                "total": 0,
                "truncated": 0,
                "truncated_pct": 0.0,
                "avg_tokens": 0.0,
                "p50_tokens": 0.0,
                "p90_tokens": 0.0,
                "p99_tokens": 0.0,
                "avg_over_by": 0.0,
            }
            continue

        # Optional sampling to keep it fast on large datasets
        if sample_size is not None and sample_size < n:
            indices = rng.sample(range(n), k=sample_size)
        else:
            indices = list(range(n))

        lengths: List[int] = []
        for idx in indices:
            text = ds[idx]["text"]
            # No truncation here: we want the true length
            ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            lengths.append(len(ids))

        arr = np.asarray(lengths, dtype=np.int32)
        truncated_mask = arr > max_length
        truncated_count = int(truncated_mask.sum())
        total = int(arr.size)

        over_by = arr[truncated_mask] - max_length
        avg_over_by = float(over_by.mean()) if over_by.size else 0.0

        out[split_name] = {
            "total": float(total),
            "truncated": float(truncated_count),
            "truncated_pct": float(100.0 * truncated_count / max(total, 1)),
            "avg_tokens": float(arr.mean()),
            "p50_tokens": float(np.percentile(arr, 50)),
            "p90_tokens": float(np.percentile(arr, 90)),
            "p99_tokens": float(np.percentile(arr, 99)),
            "avg_over_by": avg_over_by,
        }

    return out


def tokenize_datasets(
        datasets: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
) -> DatasetDict:
    max_length = min(max_length, tokenizer.model_max_length)

    def _tokenize(batch: Dict[str, List[str]]) -> BatchEncoding:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = datasets.map(_tokenize, batched=True, remove_columns=["text"])
    # Make sure label is float32 for regression (datasets 4.4.1 expects a Feature, not a string)
    tokenized = tokenized.cast_column("label", Value("float32"))
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
        gradient_accumulation_steps: int = 1,
        use_gradient_checkpointing: bool = False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        weight_decay=0.01,
        warmup_ratio=0.05,
        only_good: bool = False
) -> None:
    """Train a regression-based scoring model on the law judge dataset.

    - model_name: HF model checkpoint name.
    - use_reference: whether to include reference answers in the input.
    - only_good: if True, train ONLY on answers with high scores (ablation study).
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_dir = Path(output_dir)
    
    # --- MODIFICATION: Separate folder for ablation study ---
    if only_good:
        output_dir = Path("models") / "law_llm_advisor_only_good"
    # ------------------------------------------------------
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    # --- MODIFICATION: Pass only_good flag ---
    examples = load_all_examples(PROCESSED_DIR, only_good=only_good)

    # 2. Split by question
    train_ex, val_ex, test_ex = split_by_question(examples)

    # 3. Build HF datasets
    datasets = to_hf_dataset(train_ex, val_ex, test_ex, use_reference=use_reference)

    # If requested via CLI, you can compute truncation stats before training.
    # (This function does not change training behavior; it is just diagnostic.)

    # 4. Load tokenizer & model
    print(f"[MODEL] Loading base model: {model_name}")

    # If the user requests long sequences (e.g., 1024) but keeps a 512-token model like distilroberta,
    # training can crash due to positional embedding limits. We can auto-switch to a long-context model.
    requested_max_length = int(max_length)

    # Peek tokenizer first to estimate supported length. Some tokenizers report huge model_max_length,
    # so we will later confirm using model.config.max_position_embeddings as the source of truth.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model next (needed to check positional embedding limits reliably)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # regression
        problem_type="regression",  # hint to HF
    )

    supported_max = getattr(model.config, "max_position_embeddings", None)
    if supported_max is None:
        # Fall back to tokenizer's max length (may be huge for some tokenizers, but better than nothing)
        supported_max = int(getattr(tokenizer, "model_max_length", requested_max_length))

    # Auto-switch for convenience if the user requested a longer max_length than the selected model can handle.
    # We only do this automatically when the user is using the DEFAULT_MODEL_NAME (distilroberta-base),
    # because long-context models are heavier and should not be forced in other cases.
    if (
            AUTO_SWITCH_TO_LONG_CONTEXT
            and requested_max_length > int(supported_max)
            and model_name == DEFAULT_MODEL_NAME
    ):
        print(
            f"[MODEL] Requested max_length={requested_max_length}, but '{model_name}' supports ~{supported_max}. "
            f"Auto-switching to long-context model: {DEFAULT_LONG_CONTEXT_MODEL_NAME}"
        )
        model_name = DEFAULT_LONG_CONTEXT_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )
        supported_max = getattr(model.config, "max_position_embeddings", requested_max_length)

    # If requested_max_length is above what the model supports, cap it to avoid indexing errors.
    if requested_max_length > int(supported_max):
        print(
            f"[WARN] Requested max_length={requested_max_length} exceeds model positional limit ({supported_max}). "
            f"Capping max_length to {supported_max}."
        )
        max_length = int(supported_max)
    else:
        max_length = requested_max_length

    # Optionally enable gradient checkpointing to trade compute for memory
    if use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("[MODEL] Gradient checkpointing enabled.")
        except Exception as e:
            print(f"[MODEL] Could not enable gradient checkpointing: {e}")

    # 5. Tokenize
    tokenized = tokenize_datasets(datasets, tokenizer, max_length=max_length)

    # 6. Training args (only use arguments supported by older transformers)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        seed=RANDOM_SEED,
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
    parser = argparse.ArgumentParser(
        description="Train SuitsLLM law advisor (encoder regression judge) and optionally report truncation stats."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base HF model checkpoint to fine-tune. Default: {DEFAULT_MODEL_NAME}. Note: if you request --max-length > 512 with the default model, the script will auto-switch to {DEFAULT_LONG_CONTEXT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--use-reference",
        action="store_true",
        help="Include reference answers in the input text (recommended).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs. Default: 3.0",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate. Default: 2e-5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size. Default: 8",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length (tokens). Use 512 for short-context models; use 1024+ with a long-context model like BigBird/Longformer. Default: 512",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps. Default: 4",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )

    parser.add_argument(
        "--token-truncate",
        action="store_true",
        help=(
            "Diagnostic mode: report how many samples would be truncated at --max-length. "
            "Does not train unless you also pass --train."
        ),
    )
    parser.add_argument(
        "--token-truncate-sample",
        type=int,
        default=None,
        help="Optional: sample size per split for truncation stats (faster on large datasets).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually run training. If omitted, the script will only run diagnostics if requested.",
    )
    
    parser.add_argument(
        "--only-good", 
        action="store_true",
        help="Train only on correct answers (ablation study)"
    )

    args = parser.parse_args()

    # Always load examples/splits once if diagnostics are requested.
    if args.token_truncate:
        # Load data and build the text dataset to measure lengths.
        # --- MODIFICATION: Pass only_good flag ---
        examples = load_all_examples(PROCESSED_DIR, only_good=args.only_good)
        train_ex, val_ex, test_ex = split_by_question(examples)
        ds = to_hf_dataset(train_ex, val_ex, test_ex, use_reference=args.use_reference)

        print(f"[MODEL] Loading tokenizer for: {args.model}")
        # Match training behavior: if user asks for >512 with the default model, measure with the long-context model.
        diag_model = args.model
        if AUTO_SWITCH_TO_LONG_CONTEXT and args.max_length > 512 and diag_model == DEFAULT_MODEL_NAME:
            print(
                f"[MODEL] Diagnostic requested max_length={args.max_length} with '{diag_model}'. "
                f"Switching tokenizer to long-context model for diagnostics: {DEFAULT_LONG_CONTEXT_MODEL_NAME}"
            )
            diag_model = DEFAULT_LONG_CONTEXT_MODEL_NAME
        tok = AutoTokenizer.from_pretrained(diag_model)
        if diag_model != args.model:
            print(f"[MODEL] Using tokenizer: {diag_model}")

        stats = compute_truncation_stats(
            datasets=ds,
            tokenizer=tok,
            max_length=args.max_length,
            sample_size=args.token_truncate_sample,
            seed=RANDOM_SEED,
        )

        print("\n[TRUNCATION] Token length vs max_length diagnostics")
        print(f"  max_length = {args.max_length}")
        if args.token_truncate_sample is not None:
            print(f"  sample_size per split = {args.token_truncate_sample}")

        for split, s in stats.items():
            print(
                f"  - {split}: total={int(s['total'])}, "
                f"truncated={int(s['truncated'])} ({s['truncated_pct']:.2f}%), "
                f"avg_tokens={s['avg_tokens']:.1f}, "
                f"p50={s['p50_tokens']:.0f}, p90={s['p90_tokens']:.0f}, p99={s['p99_tokens']:.0f}, "
                f"avg_over_by={s['avg_over_by']:.1f}"
            )

        # If user didn't request training, exit here.
        if not args.train:
            raise SystemExit(0)

    # Default behavior: train (matches previous hardcoded defaults, but configurable)
    train_law_llm_advisor(
        model_name=args.model,
        use_reference=args.use_reference,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        use_gradient_checkpointing=args.grad_checkpoint,
        only_good=args.only_good # --- MODIFICATION ---
    )