"""
Assignment 1 - Optional Extension: Data Quality Analysis
===========================================================
Addresses Section 8's optional extension: "Perform data quality analysis,
e.g., token length distributions or domain coverage."

Reads the cleaned JSONL file (word-level stats -- no tokenizer needed, so
this runs instantly even on 300k+ documents) and reports, per domain:
  - document count and % share of the corpus
  - average / min / max word count per document
  - a coarse length-bucket histogram

Usage:
    python 05_optional_analysis.py --in_path ./data/cleaned.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict


def bucket(word_count: int) -> str:
    if word_count < 100:
        return "<100"
    elif word_count < 300:
        return "100-299"
    elif word_count < 700:
        return "300-699"
    elif word_count < 1500:
        return "700-1499"
    else:
        return "1500+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="./data/cleaned.jsonl")
    args = parser.parse_args()

    domain_counts = Counter()
    word_lengths = defaultdict(list)
    length_buckets = defaultdict(Counter)

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            domain = record.get("domain", "unknown")
            n_words = len(record["text"].split())
            domain_counts[domain] += 1
            word_lengths[domain].append(n_words)
            length_buckets[domain][bucket(n_words)] += 1

    total_docs = sum(domain_counts.values())

    print(f"{'='*70}\nDOMAIN COVERAGE\n{'='*70}")
    print(f"{'Domain':<12}{'Docs':>10}{'% of corpus':>14}")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_docs
        print(f"{domain:<12}{count:>10,}{pct:>13.1f}%")
    print(f"{'TOTAL':<12}{total_docs:>10,}{100.0:>13.1f}%")

    print(f"\n{'='*70}\nWORD-COUNT STATISTICS PER DOMAIN\n{'='*70}")
    print(f"{'Domain':<12}{'Avg words':>12}{'Min':>8}{'Max':>10}")
    for domain, lengths in word_lengths.items():
        avg = sum(lengths) / len(lengths)
        print(f"{domain:<12}{avg:>12.1f}{min(lengths):>8}{max(lengths):>10,}")

    print(f"\n{'='*70}\nLENGTH-BUCKET HISTOGRAM (% of each domain's documents)\n{'='*70}")
    bucket_order = ["<100", "100-299", "300-699", "700-1499", "1500+"]
    header = f"{'Domain':<12}" + "".join(f"{b:>12}" for b in bucket_order)
    print(header)
    for domain in domain_counts:
        row = f"{domain:<12}"
        n = domain_counts[domain]
        for b in bucket_order:
            pct = 100 * length_buckets[domain].get(b, 0) / n
            row += f"{pct:>11.1f}%"
        print(row)


if __name__ == "__main__":
    main()
