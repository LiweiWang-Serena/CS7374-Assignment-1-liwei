"""
Assignment 1 - Step 2: Cleaning and Normalization
===================================================
Reads raw JSONL files produced by 01_data_collection.py and produces a
single cleaned JSONL file.

Cleaning steps (per assignment Section 4.1):
  - Remove duplicate documents (exact hash dedup + near-duplicate via
    normalized-text hashing)
  - Normalize text: lowercase, collapse whitespace, strip irrelevant symbols
  - Remove very short documents (< MIN_WORDS words)
  - Strip HTML tags, markdown syntax, and reference markers (e.g. "[12]")

This script has NO external dependencies beyond the Python standard
library, so it runs anywhere (including offline sandboxes) for testing.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_WORDS = 50

# --- Regex patterns for cleaning -------------------------------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")   # [text](url) -> text
MARKDOWN_EMPHASIS_RE = re.compile(r"[*_]{1,3}")               # *bold*, _italic_, etc.
MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
REFERENCE_MARKER_RE = re.compile(r"\[\d+\]")                  # e.g. citation markers [12]
MULTI_WHITESPACE_RE = re.compile(r"\s+")
IRRELEVANT_SYMBOLS_RE = re.compile(r"[^\w\s.,!?;:'\"()%-]")   # keep basic punctuation


def clean_text(text: str) -> str:
    """Apply the full normalization pipeline to a single document."""
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_HEADER_RE.sub("", text)
    text = MARKDOWN_EMPHASIS_RE.sub("", text)
    text = REFERENCE_MARKER_RE.sub("", text)
    text = IRRELEVANT_SYMBOLS_RE.sub(" ", text)
    text = text.lower()
    text = MULTI_WHITESPACE_RE.sub(" ", text).strip()
    return text


def doc_hash(text: str) -> str:
    """Hash of the normalized text, used for exact + near-duplicate detection.

    Using the *cleaned* text (rather than raw) means documents that differ
    only in formatting/whitespace/case are correctly caught as duplicates.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def clean_file(in_path: Path, seen_hashes: set, min_words: int = MIN_WORDS):
    """Generator that yields cleaned records from a single raw JSONL file."""
    kept, dropped_short, dropped_dup, dropped_empty = 0, 0, 0, 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped_empty += 1
                continue

            raw_text = record.get("text", "")
            domain = record.get("domain", "unknown")
            cleaned = clean_text(raw_text)

            if not cleaned:
                dropped_empty += 1
                continue

            if word_count(cleaned) < min_words:
                dropped_short += 1
                continue

            h = doc_hash(cleaned)
            if h in seen_hashes:
                dropped_dup += 1
                continue
            seen_hashes.add(h)

            kept += 1
            yield {"text": cleaned, "domain": domain, "hash": h}

    stats = {
        "kept": kept,
        "dropped_short": dropped_short,
        "dropped_duplicate": dropped_dup,
        "dropped_empty_or_malformed": dropped_empty,
    }
    print(f"[clean] {in_path.name}: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="./data")
    parser.add_argument("--out_path", type=str, default="./data/cleaned.jsonl")
    parser.add_argument("--min_words", type=int, default=MIN_WORDS)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    raw_files = sorted(in_dir.glob("raw_*.jsonl"))
    if not raw_files:
        raise SystemExit(f"No raw_*.jsonl files found in {in_dir}. Run 01_data_collection.py first.")

    seen_hashes = set()
    all_stats = {}
    n_written = 0

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for raw_path in raw_files:
            for record in clean_file(raw_path, seen_hashes, args.min_words):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\n=== Cleaning complete: {n_written} documents written to {args.out_path} ===")


if __name__ == "__main__":
    main()
