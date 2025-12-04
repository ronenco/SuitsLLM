import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path

ARCHIVE_FILE_VERSION = "20250930"
MIN_QUESTION_LEN = 40
MIN_ANSWER_LEN = 60
MAX_SAMPLES = 2000

ARCHIVE_FILE_NAME = "law.stackexchange.com.7z"
DOWNLOAD_URL = f"https://archive.org/download/stackexchange_{ARCHIVE_FILE_VERSION}/stackexchange_{ARCHIVE_FILE_VERSION}/{ARCHIVE_FILE_NAME}"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
ARCHIVE_FILE = RAW_DIR / ARCHIVE_FILE_NAME
POSTS_XML = EXTRACTED_DIR / "Posts.xml"
OUTPUT_FILE = DATA_DIR / "processed" / "law_qa_pairs.jsonl"


class PostTypeId(Enum):
    QUESTION = "1"
    ANSWER = "2"



def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n  → {dest}")

    try:
        subprocess.run(["wget", "-O", str(dest), url], check=True)
    except Exception as e:
        print("ERROR: Failed to download file.")
        print("Details:", e)
        sys.exit(1)

    print("✅ Download complete.")


def extract_archive(archive_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} → {extract_to}")

    try:
        subprocess.run(["7z", "x", str(archive_path), f"-o{extract_to}"], check=True)
    except Exception:
        print("ERROR: Extraction failed. Make sure 7zip is installed")
        sys.exit(1)

    print("Extraction complete.")


def parse_posts_xml(xml_file: Path, out_file: Path, max_examples=2000):
    print(f"Streaming parse: {xml_file}")

    # Pass 1: Collect accepted answer IDs from questions
    accepted_ids = set()
    print("Pass 1: Collecting accepted answer IDs...")
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if (elem.tag == "row" and
                elem.attrib.get("PostTypeId") == PostTypeId.QUESTION.value and
                "AcceptedAnswerId" in elem.attrib):
            accepted_ids.add(elem.attrib["AcceptedAnswerId"])
        elem.clear()
    print(f"Collected {len(accepted_ids)} accepted answer IDs.")

    # Pass 2: Extract answers (only accepted ones)
    print("Pass 2: Extracting accepted answers...")
    answers = {}
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if (elem.tag == "row" and
                elem.attrib.get("PostTypeId") == PostTypeId.ANSWER.value):
            answer_id = elem.attrib.get("Id")
            if answer_id in accepted_ids:
                answers[answer_id] = elem.attrib.get("Body", "").strip()
        elem.clear()
    print(f"Stored {len(answers)} accepted answers.")

    # Pass 3: Extract questions + map to accepted answer text
    print("Pass 3: Extracting question-answer pairs...")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_file.open("w", encoding="utf-8") as f_out:
        for event, elem in ET.iterparse(xml_file, events=("end",)):
            if elem.tag != "row":
                elem.clear()
                continue

            if elem.attrib.get("PostTypeId") == PostTypeId.QUESTION.value:
                acc_id = elem.attrib.get("AcceptedAnswerId")

                if acc_id and acc_id in answers:
                    question_body = elem.attrib.get("Body", "").strip()
                    answer_body = answers[acc_id]

                    if len(question_body) >= MIN_QUESTION_LEN and len(answer_body) >= MIN_ANSWER_LEN:
                        record = {
                            "question": question_body,
                            "reference_answer": answer_body,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1

                        if written >= max_examples:
                            break

            elem.clear()

    print(f"Extracted {written} Q&A pairs into {out_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from Law StackExchange Posts.xml"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Law StackExchange dump before parsing",
    )

    args = parser.parse_args()

    # 1) If --download flag: download the file
    if args.download:
        download_file(DOWNLOAD_URL, ARCHIVE_FILE)

    # 2) Ensure archive file exists
    if not ARCHIVE_FILE.exists():
        print("\nERROR: Archive file not found:")
        print(f"   {ARCHIVE_FILE}\n")
        print("Please download it using:")
        print(f"   python {Path(__file__).name} --download\n")
        sys.exit(1)

    # 3) Extract Posts.xml if missing
    if not POSTS_XML.exists():
        extract_archive(ARCHIVE_FILE, EXTRACTED_DIR)

    # 4) Ensure Posts.xml now exists
    if not POSTS_XML.exists():
        print("\nERROR: Posts.xml not found even after extraction.")
        print("Check the archive or extraction folder.")
        sys.exit(1)

    # 5) Parse Posts.xml
    parse_posts_xml(POSTS_XML, OUTPUT_FILE, max_examples=MAX_SAMPLES)


if __name__ == "__main__":
    main()
