import argparse
import json
from pathlib import Path
from time import sleep

import httpx
from tqdm import tqdm


OLLAMA_URL = "http://localhost:11434"
MODEL_TIMEOUT = 120.0
GOOD_ANSWER_TEMPERATURE = 0.2
# System prompts for the two types of answers
GOOD_SYSTEM_PROMPT = (
    "You are a careful expert in U.S. law. "
    "Answer the question as accurately, conservatively, and clearly as possible. "
    "If the answer depends on jurisdiction or missing details, state that explicitly. "
    "Do NOT invent facts or case law. If you don't know, say that you don't know."
)
BAD_ANSWER_TEMPERATURE = 0.9
BAD_SYSTEM_PROMPT = (
    "Answer the question about U.S. law in a way that sounds confident and plausible, "
    "but includes some incorrect, misleading, oversimplified, or incomplete statements. "
    "Do NOT say that the answer is incorrect. Do NOT include any warning or disclaimer."
)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "law_qa_pairs_clean.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "law_llm_answers.jsonl"


def generate_ollama_answer(
        base_url: str,
        model: str,
        question: str,
        system_prompt: str,
        temperature: float = 0.3,
) -> str:
    """
    Call a local Ollama model via HTTP /api/chat.

    Requires:
      - ollama serve  (daemon running)
      - ollama pull <model>
    """
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        # not all models respect this, but it's here for completeness
        "options": {"temperature": temperature},
    }

    try:
        resp = httpx.post(url, json=payload, timeout=MODEL_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    data = resp.json()
    # Non-streaming chat result: { "message": { "role": "...", "content": "..." }, ... }
    msg = data.get("message", {})
    content = msg.get("content", "")
    return content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate good and bad LLM answers for law Q&A pairs using Ollama."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name (must be pulled already, e.g. llama3, mistral, qwen2).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_URL,
        help="Base URL of the Ollama server.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of Q&A pairs to process (for testing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Output JSONL file for LLM answers.",
    )

    args = parser.parse_args()

    output_file = Path(args.output)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Support resuming: count existing lines in output
    existing_lines = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for _ in f:
                existing_lines += 1
        print(
            f"Found existing output with {existing_lines} lines. "
            f"New generations will be appended."
        )

    # Load input items
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if args.max_examples is not None:
        items = items[: args.max_examples]

    start_idx = existing_lines  # one record per question
    if start_idx >= len(items):
        print("All examples already processed. Nothing to do.")
        return

    print(
        f"Generating answers for {len(items) - start_idx} "
        f"of {len(items)} total examples, using Ollama model: {args.model}"
    )

    with output_file.open("a", encoding="utf-8") as fout:
        for i in tqdm(range(start_idx, len(items)), desc="Generating", unit="q"):
            item = items[i]
            question = item["question"]
            reference_answer = item["reference_answer"]

            try:
                # Good (reliable) answer
                good_answer = generate_ollama_answer(
                    base_url=args.ollama_url,
                    model=args.model,
                    question=question,
                    system_prompt=GOOD_SYSTEM_PROMPT,
                    temperature=GOOD_ANSWER_TEMPERATURE,
                )

                # Bad (unreliable) answer
                bad_answer = generate_ollama_answer(
                    base_url=args.ollama_url,
                    model=args.model,
                    question=question,
                    system_prompt=BAD_SYSTEM_PROMPT,
                    temperature=BAD_ANSWER_TEMPERATURE,
                )

            except Exception as e:
                print(f"\nError on example {i}: {e}")
                print("Skipping this example.\n")
                continue

            record = {
                "question": question,
                "reference_answer": reference_answer,
                "llm_answer_correct": good_answer,
                "llm_answer_incorrect": bad_answer,
                "source_model": args.model,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved LLM answers to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
