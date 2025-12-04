import html
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE =  PROCESSED_DIR / "law_qa_pairs.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "law_qa_pairs_clean.jsonl"


def clean_html(html_text: str) -> str:
    # Unescape HTML entities (&amp; → &, &quot; → ")
    html_text = html.unescape(html_text)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")

    # Replace <code> with backticks for readability
    for code_block in soup.find_all("code"):
        code_block.string = f"`{code_block.get_text(strip=True)}`"

    # Replace blockquotes with prefix "> "
    for bq in soup.find_all("blockquote"):
        quote_text = bq.get_text("\n", strip=True)
        bq.replace_with("\n> " + quote_text + "\n")

    # Extract visible text with normalized newlines
    text = soup.get_text("\n", strip=True)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cleaning HTML from: {INPUT_FILE}")
    print(f"Saving cleaned file to: {OUTPUT_FILE}")

    count = 0
    with INPUT_FILE.open("r", encoding="utf-8") as fin, \
            OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)

            item["question"] = clean_html(item["question"])
            item["reference_answer"] = clean_html(item["reference_answer"])

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Cleaned {count} Q&A items.")


if __name__ == "__main__":
    main()
