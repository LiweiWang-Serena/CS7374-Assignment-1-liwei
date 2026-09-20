# Assignment 1: Data Collection and Preprocessing for Foundation Model Pre-Training

A multi-domain (encyclopedic + news + general web text) text pipeline that
collects, cleans, tokenizes, and batches data for transformer pretraining.

```
collect (3 domains) -> clean (dedup/filter/normalize) -> tokenize (GPT-2 BPE, chunked)
   -> custom PyTorch DataLoader (map-style AND streaming)
```

## Project structure

```
.
├── scripts/
│   ├── 01_data_collection.py   # Step 1: download raw text from 3 domains
│   ├── 02_cleaning.py          # Step 2: dedup, filter, normalize (stdlib only)
│   ├── 03_tokenization.py      # Step 3: GPT-2 BPE tokenize + chunk to blocks
│   ├── 04_dataloader.py        # Step 4: PyTorch Dataset/IterableDataset + DataLoader
│   └── 05_optional_analysis.py # Optional (Section 8): domain coverage + word-length stats
├── data/                       # created by the pipeline (raw + cleaned + tokenized)
├── sample_data/                # created by Step 3 (first N tokenized blocks, .pt)
├── requirements.txt
├── Assignment1_Report.docx     # written report (Section 5.3 of the assignment)
└── README.md                   # this file
```

## Setup

Requires internet access (Steps 1 and 3 download from the Hugging Face Hub) and
Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Gotcha:** `source venv/bin/activate` only applies to the *current terminal
session*. Opening a new terminal tab/window resets it — if you see
`ModuleNotFoundError: No module named 'torch'`, re-run the activate command
before anything else.

## Running the pipeline

Run these four commands from the project root, in order.

### Step 1 — Data collection

```bash
python3 scripts/01_data_collection.py --target_gb 1.2 --out_dir ./data
```

Streams from three Hugging Face datasets until ~1.2GB of raw text is
collected, split roughly 60/25/15 across domains:

| Domain | Source | Config |
|---|---|---|
| Wikipedia (encyclopedic) | `wikimedia/wikipedia` | `20231101.en` |
| CC-News (news) | `vblagoje/cc_news` | — |
| OpenWebText (web) | `Skylion007/openwebtext` | — |

Writes `raw_wikipedia.jsonl`, `raw_news.jsonl`, `raw_web.jsonl`, and
`collection_manifest.json` (doc counts + byte totals per domain) to `--out_dir`.

**Gotcha:** Hugging Face `datasets` no longer supports Python-script-based
dataset loading. Bare identifiers without a namespace (e.g. the old `cc_news`)
will fail with `RuntimeError: Dataset scripts are no longer supported` — use
the namespaced, Parquet-format mirrors listed above. If a source ever moves
again, fallbacks are noted as comments at the top of `01_data_collection.py`.
The script won't let one failed domain take down the other two: it catches
per-domain errors, prints a fallback hint, and keeps going.

### Step 2 — Cleaning and normalization

```bash
python3 scripts/02_cleaning.py --in_dir ./data --out_path ./data/cleaned.jsonl --min_words 50
```

Pure standard-library (no third-party deps): strips HTML/markdown/citation
markers, lowercases, collapses whitespace, drops documents under
`--min_words`, and removes exact duplicates via SHA-256 hashing of the
*normalized* text (so formatting-only differences still count as duplicates).
Prints per-source stats (`kept` / `dropped_short` / `dropped_duplicate` /
`dropped_empty_or_malformed`) — save these for the report.

### Step 3 — Tokenization

```bash
python3 scripts/03_tokenization.py \
  --in_path ./data/cleaned.jsonl \
  --out_path ./data/tokenized_blocks.pt \
  --sample_out ./sample_data/sample_dataset.pt \
  --tokenizer_name gpt2 \
  --block_size 512 \
  --n_sample_blocks 10
```

Tokenizes with GPT-2's BPE tokenizer, chunks any document longer than
`--block_size` into non-overlapping blocks (pass `--stride` for overlapping
chunks instead), pads the final chunk with EOS, and saves the full dataset
plus a small sample (this sample file is the Section 5.2 deliverable).

**Gotcha:** you'll see `Token indices sequence length is longer than the
specified maximum sequence length for this model (N > 1024)` repeatedly —
this is benign. It assumes you're about to feed the sequence straight into
the GPT-2 *model*; we don't — we chunk it ourselves right after. No tokens
are lost or truncated.

**Gotcha:** both output paths' parent directories are auto-created, so this
won't crash even on a fresh checkout — but if you're running an older copy of
this script, make sure `--out_dir` / `sample_data/` exist first
(`mkdir -p data sample_data`).

### Step 4 — DataLoader smoke test

```bash
# map-style: loads the whole tokenized tensor into memory, supports shuffling
python3 scripts/04_dataloader.py --mode map --tokenized_path ./data/tokenized_blocks.pt --batch_size 8

# streaming: reads cleaned.jsonl line-by-line, tokenizes on the fly, never
# materializes the full dataset in memory (for corpora too large for RAM)
python3 scripts/04_dataloader.py --mode stream --jsonl_path ./data/cleaned.jsonl --batch_size 8
```

Both print a few `batch N: input_ids=(8, 512) attention_mask=(8, 512)` lines
if everything is wired correctly.

### Step 5 (optional, Section 8 extension) — Data quality analysis

```bash
python3 scripts/05_optional_analysis.py --in_path ./data/cleaned.jsonl
```

Pure stdlib, runs in seconds even on 300k+ documents. Reports domain
coverage and word-count distribution — directly addresses Section 8's
"perform data quality analysis, e.g., token length distributions or domain
coverage."

## Actual results from a verified run

| Stage | Result |
|---|---|
| Raw collected | 357,171 docs / 1.288 GB (Wikipedia 174,374 · News 144,080 · Web 38,717) |
| After cleaning | 317,064 docs kept (88.8%) — News lost the most to dedup (14,104 dup docs, 9.8%); Wikipedia had 0 duplicates |
| Tokenized | 656,119 blocks of 512 tokens (Wikipedia 398,928 · News 165,254 · Web 91,937), tensor shape `(656119, 512)` |
| Sample file | `sample_data/sample_dataset.pt`, shape `(10, 512)` |
| DataLoader | both map-style and streaming modes verified producing correctly-shaped `(8, 512)` batches |
| Domain coverage | Wikipedia 52.2% · News 35.6% · Web 12.2% of cleaned corpus |
| Word length | Wikipedia avg 737.9 (max 34,909) · News avg 423.1 (max 17,455) · Web avg 829.7 (max 18,903) |

See `Assignment1_Report.docx` for full discussion, reasoning, and reflections.

## Submission checklist (Assignment Section 6)

- [ ] `scripts/*.py` — the four pipeline scripts
- [ ] `sample_data/sample_dataset.pt` — sample tokenized batches
- [ ] `Assignment1_Report.pdf` — export `Assignment1_Report.docx` to PDF, fill in your name first
