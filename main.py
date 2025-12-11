"""
SuitsLLM CLI entry point.

Flow:
  1. Choose LLM backend: llama / gemini / chatgpt / best-of-all.
  2. The chosen LLM(s) "load" (in practice: we call a backend interface; currently mocked).
  3. Ask the user for a legal question (prompt).
  4. Each selected LLM answers the question with a single-shot prompt, including a fixed system instruction:
     "You are a law helper..."
  5. Load the advisor model from models/law_llm_advisor.
  6. The advisor scores each answer with a continuous quality score in [0, 1].
  7. Print to the screen: the answer and its score (0 = worst, 1 = best).
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import httpx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional external providers (OpenAI, Gemini)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

# ---------------------------
# Configuration
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ADVISOR_MODEL_DIR = PROJECT_ROOT / "models" / "law_llm_advisor"

# Local Ollama configuration (for real LLM answers)
OLLAMA_URL = "http://localhost:11434"
MODEL_TIMEOUT = 120.0

# Map our logical backends to Ollama model names (used for the "llama" backend).
# You can change these to any models you have pulled with `ollama pull <model>`.
DEFAULT_OLLAMA_MODEL = "mistral"
BACKEND_TO_OLLAMA_MODEL = {
    "llama": "mistral",   # e.g. "llama3" if you have it
}

# Remote provider configuration (OpenAI & Gemini)
OPENAI_KEY_ENV_VAR = "OPENAI_API_KEY"
GEMINI_KEY_ENV_VAR = "GEMINI_API_KEY"

# Default chat models; can be overridden via environment variables if desired.
OPENAI_CHAT_MODEL = os.getenv("SUITSLLM_OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_CHAT_MODEL = os.getenv("SUITSLLM_GEMINI_MODEL", "gemini-2.5-flash")

AVAILABLE_LLMS = ["llama", "gemini", "chatgpt"]

SYSTEM_PROMPT = (
    "You are a helpful legal assistant. "
    "You help users reason about law-related questions, explain relevant concepts clearly, "
    "and avoid giving definitive legal advice.\n"
    "Answer concisely and in plain language. "
    "If something is uncertain or jurisdiction-specific, say so explicitly."
)


# ---------------------------
# Utility: device selection
# ---------------------------

def get_device() -> torch.device:
    """
    Pick the best available device: MPS (Apple), CUDA, or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Advisor model wrapper
# ---------------------------

class LawLLMAdvisor:
    """
    Wrapper around the trained advisor model in models/law_llm_advisor.

    The model is a regression model that outputs a single continuous score
    in [0, 1], representing answer quality.
    """

    def __init__(self, model_dir: Path, max_length: int = 512) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise RuntimeError(
                f"Advisor model directory not found: {self.model_dir}. "
                f"Make sure you have trained and saved the model under models/law_llm_advisor."
            )

        self.device = get_device()
        print(f"[ADVISOR] Loading advisor model from {self.model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @staticmethod
    def build_input_text(question: str, answer: str) -> str:
        """
        Build the input text for the advisor model.

        NOTE: This should mirror (as closely as possible) the formatting used during training.
        Here we use the question + student's answer layout without a reference answer.
        """
        return (
            "You are a law professor grading student answers.\n"
            "Consider the question and the student's answer.\n\n"
            "[QUESTION]\n"
            f"{question.strip()}\n\n"
            "[STUDENT_ANSWER]\n"
            f"{answer.strip()}"
        )

    def score(self, question: str, answer: str) -> float:
        """
        Score a (question, answer) pair.

        Returns:
            score  (float): continuous quality score, roughly in [0, 1].
        """
        text = self.build_input_text(question, answer)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, 1) for regression

        # Squeeze to scalar and clamp to [0, 1] for presentation
        raw_score = float(logits.squeeze().detach().cpu().numpy())
        score = float(np.clip(raw_score, 0.0, 1.0))
        return score


# ---------------------------
# External LLM client helpers
# ---------------------------

_openai_client: Optional[Any] = None
_gemini_initialized: bool = False


def get_openai_client() -> "OpenAI":
    """
    Lazily initialize and return an OpenAI client using OPENAI_API_KEY.

    Requires:
      - openai package installed
      - OPENAI_API_KEY environment variable set (or equivalent)
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if OpenAI is None:
        raise RuntimeError(
            "OpenAI client requested but 'openai' package is not installed.\n"
            "Install with: pip install openai"
        )

    api_key = os.getenv(OPENAI_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing OpenAI API key. Please set {OPENAI_KEY_ENV_VAR} in your environment."
        )

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def ensure_gemini_initialized() -> None:
    """
    Lazily configure the Gemini client using GEMINI_API_KEY.
    """
    global _gemini_initialized
    if _gemini_initialized:
        return

    if genai is None:
        raise RuntimeError(
            "Gemini requested but 'google-generativeai' package is not installed.\n"
            "Install with: pip install google-generativeai"
        )

    api_key = os.getenv(GEMINI_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing Gemini API key. Please set {GEMINI_KEY_ENV_VAR} in your environment."
        )

    genai.configure(api_key=api_key)
    _gemini_initialized = True


def generate_openai_answer(question: str) -> str:
    """
    Generate an answer using OpenAI Chat Completions API (chatgpt backend).
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def generate_gemini_answer(question: str) -> str:
    """
    Generate an answer using Google Gemini API (gemini backend).
    """
    ensure_gemini_initialized()
    # genai is configured globally; we use a GenerativeModel instance
    model_obj = genai.GenerativeModel(GEMINI_CHAT_MODEL)
    response = model_obj.generate_content(
        [
            SYSTEM_PROMPT,
            question.strip(),
        ],
        generation_config={"temperature": 0.2},
    )
    content = getattr(response, "text", "") or ""
    return content.strip()


# ---------------------------
# LLM backend helpers
# ---------------------------

def generate_ollama_answer(
        base_url: str,
        model: str,
        question: str,
        system_prompt: str,
        temperature: float = 0.3,
) -> str:
    """Call a local Ollama model via HTTP /api/chat.

    Requires:
      - "ollama serve" running in the background
      - the requested model to be pulled via `ollama pull <model>`
    """
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "options": {"temperature": temperature},
    }

    try:
        resp = httpx.post(url, json=payload, timeout=MODEL_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    return content.strip()


def generate_with_llm(backend: str, question: str) -> str:
    """Generate an answer using the selected LLM backend.

    - "llama"  -> local Ollama model (see BACKEND_TO_OLLAMA_MODEL).
    - "gemini" -> Google Gemini API.
    - "chatgpt"-> OpenAI Chat Completions API.

    All backends use the same SYSTEM_PROMPT as the system / preamble.
    """
    backend = backend.lower()
    if backend not in AVAILABLE_LLMS:
        raise ValueError(f"Unsupported backend: {backend}")

    if backend == "llama":
        model_name = BACKEND_TO_OLLAMA_MODEL.get("llama", DEFAULT_OLLAMA_MODEL)
        try:
            answer = generate_ollama_answer(
                base_url=OLLAMA_URL,
                model=model_name,
                question=question.strip(),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate answer with backend 'llama' "
                f"(Ollama model '{model_name}'): {e}"
            ) from e
        return answer

    if backend == "gemini":
        try:
            return generate_gemini_answer(question)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate answer with backend 'gemini' "
                f"(Gemini model '{GEMINI_CHAT_MODEL}'): {e}"
            ) from e

    if backend == "chatgpt":
        try:
            return generate_openai_answer(question)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate answer with backend 'chatgpt' "
                f"(OpenAI model '{OPENAI_CHAT_MODEL}'): {e}"
            ) from e

    # This should be unreachable due to AVAILABLE_LLMS check
    raise ValueError(f"Unhandled backend: {backend}")


# ---------------------------
# CLI helpers
# ---------------------------

def choose_llm_mode() -> str:
    """
    Ask the user how to run the system:
      - single backend (llama/gemini/chatgpt)
      - best-of (run all and pick the top-scoring one)

    Returns:
        mode: one of {"llama", "gemini", "chatgpt", "best"}
    """
    print("=== SuitsLLM: Law Helper CLI ===")
    print("Choose LLM backend:")
    print("  1) llama")
    print("  2) gemini")
    print("  3) chatgpt")
    print("  4) best-of (run all and pick best answer)")
    while True:
        choice = input("Enter choice [1-4]: ").strip()
        if choice == "1":
            return "llama"
        if choice == "2":
            return "gemini"
        if choice == "3":
            return "chatgpt"
        if choice == "4":
            return "best"
        print("Invalid choice, please enter 1, 2, 3, or 4.")


def ask_user_question() -> str:
    """
    Prompt the user to type their legal question.
    """
    print("\nPlease type your law-related question. Finish with ENTER:")
    question = input("> ")
    return question.strip()


# ---------------------------
# Main CLI flow
# ---------------------------

def run_cli() -> None:
    mode = choose_llm_mode()
    question = ask_user_question()

    # Load advisor model once
    advisor = LawLLMAdvisor(ADVISOR_MODEL_DIR, max_length=512)

    if mode == "best":
        backends = AVAILABLE_LLMS
    else:
        backends = [mode]

    results: List[Tuple[str, str, float]] = []  # (backend, answer, score)

    for backend in backends:
        print(f"\n[LLM] Generating answer with backend: {backend} ...")

        # Try to generate an answer with the selected backend.
        try:
            answer = generate_with_llm(backend, question)
        except Exception as e:
            msg = str(e)
            print(f"[WARN] Backend '{backend}' failed to generate an answer: {msg}")

            # Heuristic: if this looks like a quota/limit issue, make it explicit.
            lower_msg = msg.lower()
            if (
                "insufficient_quota" in lower_msg
                or "exceeded your current quota" in lower_msg
                or "quota" in lower_msg
                or "rate limit" in lower_msg
                or "429" in lower_msg
            ):
                print(
                    f"[WARN] It looks like the quota or rate limits for backend '{backend}' "
                    f"have been exhausted or restricted. Skipping this backend."
                )

            # Skip to the next backend instead of aborting the whole run.
            continue

        print(f"\n[LLM:{backend}] Answer:\n{'-' * 40}\n{answer}\n{'-' * 40}")

        # Try to score the answer with the advisor.
        try:
            score = advisor.score(question, answer)
        except Exception as e:
            print(
                f"[WARN] Advisor failed to score the answer from backend '{backend}': {e}. "
                "Skipping this backend in the aggregated results."
            )
            continue

        results.append((backend, answer, score))
        print(f"[ADVISOR] {backend} answer scored as {score:.3f} (0 = worst, 1 = best)\n")

    if mode == "best":
        if not results:
            print(
                "\n[ERROR] No successful LLM answers were generated. "
                "Please check your configuration, API keys, or provider quotas."
            )
            return

        # Pick the answer with the highest score
        results_sorted = sorted(
            results,
            key=lambda x: x[2],  # sort by score
            reverse=True,
        )
        best_backend, best_answer, best_score = results_sorted[0]
        print("\n=== BEST ANSWER (according to advisor) ===")
        print(f"Backend: {best_backend}")
        print(f"Score:   {best_score:.3f} (0 = worst, 1 = best)")
        print(f"Answer:\n{'=' * 40}\n{best_answer}\n{'=' * 40}\n")


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)