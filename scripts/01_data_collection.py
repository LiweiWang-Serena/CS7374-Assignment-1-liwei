"""
Assignment 1 - Step 1: Data Collection
========================================
Collects multi-domain text data for foundation model pretraining.

Sources (chosen to satisfy the "diversity" requirement in Section 3):
  1. Wikipedia (encyclopedic domain)        -> wikimedia/wikipedia (20231101.en)
  2. CC-News (news domain)                  -> vblagoje/cc_news
  3. OpenWebText (general web text domain)  -> Skylion007/openwebtext

NOTE on dataset names: as of late 2025, Hugging Face `datasets` dropped
support for Python-script-based dataset loading entirely. Bare identifiers
like "cc_news" (no namespace) pointed to script-based loaders and will now
raise "Dataset scripts are no longer supported". We therefore use the
namespaced, Parquet-format mirrors above. If a source below still fails
for you (Hub datasets occasionally get re-migrated), the known-good
fallback mirrors are:
    - CC-News fallback:      vblagoje/cc_news (already primary above)
    - OpenWebText fallback:  vietgpt/openwebtext_en
    - Wikipedia fallback:    omarkamali/wikipedia-monthly, config "latest.en"
      (more current than the static 20231101 dump, if freshness matters)

NOTE: This script requires internet access. Run it in Google Colab or any
machine with an active connection (it will NOT run in an offline sandbox).

Usage:
    python 01_data_collection.py --target_gb 1.2 --out_dir ./data
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "Missing dependency. Run first:\n"
        "  pip install datasets huggingface_hub"
    )


def bytes_of(text: str) -> int:
    """UTF-8 byte size of a string (used to track how close we are to target_gb)."""
    return len(text.encode("utf-8"))


def collect_domain(
    dataset_name: str,
    split: str,
    text_field: str,
    domain_label: str,
    target_bytes: int,
    out_path: Path,
    config_name: str | None = None,
    streaming: bool = True,
):
    """
    Stream a HF dataset and write raw documents (one JSON object per line)
    to out_path until target_bytes of raw text has been collected.

    Streaming is used so we never have to download/hold the entire dataset
    (some of these corpora are hundreds of GB) -- we only pull as many
    examples as we actually need.
    """
    print(f"[collect] domain={domain_label} dataset={dataset_name} target={target_bytes/1e6:.1f}MB")

    ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=streaming,
    )

    collected_bytes = 0
    n_docs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get(text_field, "")
            if not text or not isinstance(text, str):
                continue
            record = {"text": text, "domain": domain_label}
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")
            collected_bytes += bytes_of(text)
            n_docs += 1
            if collected_bytes >= target_bytes:
                break

    print(f"[done] {domain_label}: {n_docs} docs, {collected_bytes/1e6:.1f}MB -> {out_path}")
    return n_docs, collected_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gb", type=float, default=1.2,
                         help="Total raw text volume to collect, in GB (>=1GB per assignment spec).")
    parser.add_argument("--out_dir", type=str, default="./data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_target_bytes = int(args.target_gb * (1024 ** 3))

    # Split the total budget across three domains to guarantee diversity.
    # Wikipedia gets the largest share since it's our primary source;
    # news + web text are supplementary domains to satisfy the
    # "multiple domains" requirement in Section 3.
    split_ratio = {"wikipedia": 0.6, "news": 0.25, "web": 0.15}

    # (dataset_name, config_name, out_filename, fallback_hint) per domain --
    # kept as a list so one failing source doesn't take the other two down
    # with it (Hub dataset repos occasionally get re-migrated).
    domain_specs = [
        ("wikipedia", "wikimedia/wikipedia", "20231101.en", "raw_wikipedia.jsonl",
         'fallback: load_dataset("omarkamali/wikipedia-monthly", "latest.en", streaming=True)'),
        ("news", "vblagoje/cc_news", None, "raw_news.jsonl",
         'fallback: try config_name=None removed, or open an issue on the dataset\'s HF discussion page'),
        ("web", "Skylion007/openwebtext", None, "raw_web.jsonl",
         'fallback: load_dataset("vietgpt/openwebtext_en", streaming=True)'),
    ]

    manifest = {}
    failures = []

    for domain_label, dataset_name, config_name, out_filename, fallback_hint in domain_specs:
        try:
            n, b = collect_domain(
                dataset_name=dataset_name,
                config_name=config_name,
                split="train",
                text_field="text",
                domain_label=domain_label,
                target_bytes=int(total_target_bytes * split_ratio[domain_label]),
                out_path=out_dir / out_filename,
            )
            manifest[domain_label] = {"docs": n, "bytes": b, "source": dataset_name}
        except Exception as e:
            print(f"\n[FAILED] domain={domain_label} dataset={dataset_name}")
            print(f"  error: {e}")
            print(f"  {fallback_hint}")
            failures.append(domain_label)

    with open(out_dir / "collection_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(v["bytes"] for v in manifest.values())
    print(f"\n=== TOTAL COLLECTED: {total_bytes/1e9:.3f} GB across {len(manifest)}/3 domains ===")
    if failures:
        print(f"=== {len(failures)} domain(s) FAILED: {failures} -- see fallback hints above, "
              f"or paste the error back for help debugging. ===")


if __name__ == "__main__":
    main()
