import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

# Attempt to import required libraries dynamically
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Configuration Constants ---
OPENAI_KEY_ENV_VAR = "OPENAI_API_KEY"
GEMINI_KEY_ENV_VAR = "GEMINI_API_KEY"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds

# File Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "law_llm_answers.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "law_judge_dataset.jsonl"

# Judge Model Configuration
VALID_SCORES = {0.0, 0.5, 1.0}
TEMPERATURE = 0.0
SYSTEM_PROMPT = (
    "You are an expert evaluator of legal correctness in U.S. law. "
    "Given a question, the accepted reference answer, and an LLM-generated answer, "
    "your task is to rate the LLM answer using only the following discrete scores:\n\n"
    "1.0 = fully correct\n"
    "0.5 = partially correct or incomplete\n"
    "0.0 = incorrect or misleading\n\n"
    "Output ONLY the number. No explanation."
)
JUDGE_USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Reference Answer (trusted ground truth):\n{reference_answer}\n\n"
    "LLM Answer to evaluate:\n{llm_answer}\n\n"
    "Your score (0, 0.5, or 1):"
)

# Type Aliases for Clarity
JudgeClient = Any
JudgeItem = Dict[str, Any]


def _snap_score(x: float) -> float:
    """Snaps a float score to the closest valid score (0.0, 0.5, 1.0)."""
    return min(VALID_SCORES, key=lambda a: abs(a - x))


def _parse_score(content: str, provider_name: str) -> float:
    """Parses model output to a float score and handles out-of-range values."""
    try:
        score = float(content)
    except ValueError as e:
        raise ValueError(f"{provider_name} judge returned invalid score: '{content}'") from e

    if score not in VALID_SCORES:
        print(
            f"Warning: {provider_name} judge returned out-of-range score: {score}, "
            "snapping to closest allowed score."
        )
        score = _snap_score(score)

    return score


def _get_file_paths(input_arg: str, output_arg: str) -> Dict[str, Path]:
    """Sets up input, output, and log file paths."""
    input_path = Path(input_arg)
    output_path = Path(output_arg)
    # Log file is placed in 'logs' directory sibling to 'processed'
    fail_log_path = DATA_DIR / "logs" / output_path.name.replace(".jsonl", ".log")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output and log directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_log_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "fail_log_path": fail_log_path,
    }


def judge_openai(
        client: JudgeClient,
        model: str,
        question: str,
        reference_answer: str,
        llm_answer: str,
) -> float:
    """Ask an OpenAI model to grade the LLM answer."""
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        llm_answer=llm_answer,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
    )

    content = response.choices[0].message.content.strip()
    return _parse_score(content, "OpenAI")


def judge_gemini(
        client: JudgeClient,
        model: str,
        question: str,
        reference_answer: str,
        llm_answer: str,
) -> float:
    """Judge using Google Gemini API."""
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        llm_answer=llm_answer,
    )

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    model_obj = client.GenerativeModel(model)

    response = model_obj.generate_content(
        full_prompt,
        generation_config={"temperature": TEMPERATURE}
    )

    content = response.text.strip()
    return _parse_score(content, "Gemini")


def judge_ollama(
        base_url: str,
        model: str,
        question: str,
        reference_answer: str,
        llm_answer: str,
) -> float:
    """Judge using local Ollama model."""
    url = f"{base_url.rstrip('/')}/api/chat"

    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        llm_answer=llm_answer,
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": TEMPERATURE},
    }

    resp = httpx.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()

    content = resp.json()["message"]["content"].strip()
    return _parse_score(content, "Ollama")


def judge_dispatcher(
        client: JudgeClient,
        provider: str,
        model: str,
        ollama_url: str,
        question: str,
        reference_answer: str,
        llm_answer: str,
) -> float:
    """Dispatches the judging request to the correct provider function."""
    if provider == "openai":
        return judge_openai(client, model, question, reference_answer, llm_answer)
    elif provider == "gemini":
        return judge_gemini(client, model, question, reference_answer, llm_answer)
    elif provider == "ollama":
        return judge_ollama(ollama_url, model, question, reference_answer, llm_answer)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _setup_client(provider: str, api_key_arg: Optional[str]) -> Tuple[JudgeClient, Optional[str]]:
    """Initializes the required API client based on the provider."""
    client: JudgeClient = None
    api_key: Optional[str] = None

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI library not installed. Install with 'pip install openai'.")
        api_key = api_key_arg or os.getenv(OPENAI_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key. Use --api-key or set {OPENAI_KEY_ENV_VAR}.")
        client = OpenAI(api_key=api_key)

    elif provider == "gemini":
        if genai is None:
            raise RuntimeError("Gemini support not available. Install with 'pip install google-generativeai'.")
        api_key = api_key_arg or os.getenv(GEMINI_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key. Use --api-key or set {GEMINI_KEY_ENV_VAR}.")
        genai.configure(api_key=api_key)
        # For Gemini, the client is implicitly configured, we pass genai module itself
        client = genai

    # Ollama does not need a client object, httpx is used directly
    elif provider == "ollama":
        client = None

    return client, api_key


def _count_input_lines(input_path: Path) -> int:
    """Counts total lines in the input file for TQDM without loading into memory."""
    try:
        with input_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _get_resume_count(output_path: Path) -> int:
    """Counts the number of successfully judged items in the output file for resuming."""
    existing = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            # We must count lines because we write two records per input line (correct/incorrect)
            existing = sum(1 for _ in f)
        print(f"Resuming: {existing} items already judged.")
    return existing


def run_judging_loop(
        client: JudgeClient,
        provider: str,
        model: str,
        ollama_url: str,
        sleep_time: float,
        max_examples: Optional[int],
        paths: Dict[str, Path]
) -> None:
    """The main loop for processing and judging LLM answers, iterating line-by-line."""
    input_path = paths["input_path"]
    output_path = paths["output_path"]
    fail_log_path = paths["fail_log_path"]

    total_input_lines = _count_input_lines(input_path)
    lines_to_skip = _get_resume_count(output_path)

    # Since each input line generates 2 output records, we adjust the input iterator start index
    input_start_index = lines_to_skip // 2

    # Calculate the total items TQDM should display based on the limit
    if max_examples is not None:
        total_to_process = min(total_input_lines, max_examples)
    else:
        total_to_process = total_input_lines

    if input_start_index >= total_to_process:
        print("All items processed. Exiting.")
        return

    print(
        f"Starting processing from input item {input_start_index + 1} out of {total_to_process}.")

    # Use 'islice' to skip lines read from the input file handle
    with output_path.open("a", encoding="utf-8") as fout, \
            fail_log_path.open("a", encoding="utf-8") as flog, \
            input_path.open("r", encoding="utf-8") as fin:

        # If max_examples is set, we only process up to that limit.
        stop_index = max_examples if max_examples is not None else None
        # Create an iterator that starts at the correct resume point and stops at max_examples
        input_iterator = islice(fin, input_start_index, stop_index)

        progress_bar = tqdm(input_iterator,
                            initial=input_start_index,
                            total=total_to_process,
                            unit="item",
                            desc="Judging")

        for i, line in enumerate(progress_bar):
            current_input_index = i + input_start_index

            # The max_examples check is redundant due to islice, but kept for clarity/safety.
            if max_examples is not None and current_input_index >= max_examples:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                fail_msg = f"Input item {current_input_index + 1} failed JSON decoding: {str(e)}\n"
                flog.write(fail_msg)
                print(f"\n{fail_msg.strip()}")
                continue  # Skip to the next line

            question = item.get("question", "")
            reference_answer = item.get("reference_answer", "")
            source_model = item.get("source_model", "unknown")

            # We must judge BOTH good and bad answers as separate samples
            for answer_type_key in ["llm_answer_correct", "llm_answer_incorrect"]:
                llm_answer = item.get(answer_type_key, "")

                # Map "llm_answer_correct" -> "good", "llm_answer_incorrect" -> "bad"
                type_flag = "good" if answer_type_key == "llm_answer_correct" else "bad"

                score: Optional[float] = None

                try:
                    score = judge_dispatcher(
                        client, provider, model, ollama_url,
                        question, reference_answer, llm_answer
                    )
                except Exception as e:
                    # Use absolute index in fail message
                    fail_msg = (
                        f"Input item {current_input_index + 1} ({type_flag}) failed with model '{model}': {type(e).__name__} - {str(e)}"
                    )
                    flog.write(fail_msg + "\n")
                    print(f"\n{fail_msg}")

                record: JudgeItem = {
                    "question": question,
                    "reference_answer": reference_answer,
                    "llm_answer": llm_answer,
                    "source_model": source_model,  # Model that generated the answer
                    "answer_type": type_flag,
                    "judge_model": model,  # Model used for judging
                    "label": score,
                }

                # Write the record regardless of success (score=None if failed)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Pause between judge calls
                if provider != "ollama":
                    sleep(sleep_time)

    print(f"\nDone. Saved judged dataset to: {output_path.resolve()}")
    print(f"Failures logged to: {fail_log_path.resolve()}")


def retry_failed_scores(
        client: JudgeClient,
        provider: str,
        model: str,
        ollama_url: str,
        sleep_time: float,
        paths: Dict[str, Path]
) -> None:
    """Reprocesses only entries marked as failed in the output file."""
    output_path = paths["output_path"]
    fail_log_path = paths["fail_log_path"]

    print("\nRETRY MODE ENABLED — Reprocessing failed entries only.\n")

    if not output_path.exists():
        raise RuntimeError("Cannot retry — output file does not exist yet.")

    # Read all existing data and identify failed ones
    existing_entries: List[JudgeItem] = []
    failed_entries_to_retry: List[JudgeItem] = []

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            existing_entries.append(entry)
            # We look for a null label.
            # Only retry if the item explicitly failed to get a score.
            if entry.get("label") is None:
                failed_entries_to_retry.append(entry)

    if not failed_entries_to_retry:
        print("No failed entries found to reprocess. Exiting retry mode.")
        return

    print(f"Found {len(failed_entries_to_retry)} failed entries to reprocess.")

    # We'll rebuild the entire output file
    temp_output = output_path.with_suffix(".tmp")
    retried_count = 0

    with temp_output.open("w", encoding="utf-8") as fout, \
            fail_log_path.open("a", encoding="utf-8") as flog:

        for entry in tqdm(existing_entries, desc="Retrying failed entries"):
            if not (entry.get("label") is None):
                # Write unchanged row if it was successful
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            # This is a failed entry, attempt to judge again
            retried_count += 1
            question = entry["question"]
            reference_answer = entry["reference_answer"]
            llm_answer = entry["llm_answer"]

            # Reset flags before retry
            entry["label"] = None

            try:
                score = judge_dispatcher(
                    client, provider, model, ollama_url,
                    question, reference_answer, llm_answer
                )
                entry["label"] = score
            except Exception as e:
                fail_msg = f"Retry {retried_count} failed again: {type(e).__name__} - {e}\n"
                print(f"\n{fail_msg.strip()}")
                flog.write(fail_msg)

            # Write the retried or failed entry
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if provider != "ollama":
                sleep(sleep_time)

    # Replace original file with the updated file
    temp_output.replace(output_path)

    print(f"\nRetry completed. Updated file written to {output_path.resolve()}")
    print(f"Retried {retried_count} samples.")


def main() -> None:
    """Parses arguments and runs the core logic."""
    parser = argparse.ArgumentParser(
        description="Judge correctness of LLM answers using an expert LLM judge."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "ollama"],
        required=True,
        help="Which judge model provider to use (openai, gemini, or ollama)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for chosen provider (e.g., gpt-4-turbo, gemini-2.5-flash, or llama3)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Overrides environment variables."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
        help=f"Path to the input JSONL file containing LLM answers. Default: {INPUT_FILE}"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Path to the output JSONL file to write judged results. Default: {OUTPUT_FILE}"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_URL,
        help=f"Base URL for the Ollama API. Default: {OLLAMA_URL}"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pause between judge calls (ignored for Ollama). Default: 0.2 seconds."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit total number of samples processed for debugging."
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Reprocess only entries marked as failed in the OUTPUT_FILE."
    )

    args = parser.parse_args()

    try:
        paths = _get_file_paths(args.input, args.output)
        client, _ = _setup_client(args.provider, args.api_key)
    except Exception as e:
        print(f"Setup Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.retry_failures:
        retry_failed_scores(
            client=client,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
            sleep_time=args.sleep,
            paths=paths
        )
    else:
        run_judging_loop(
            client=client,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
            sleep_time=args.sleep,
            max_examples=args.max_examples,
            paths=paths
        )


if __name__ == "__main__":
    main()
