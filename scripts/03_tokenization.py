"""
Assignment 1 - Step 3: Tokenization
=====================================
Reads the cleaned JSONL file (from 02_cleaning.py), tokenizes every
document with a transformer-compatible tokenizer, and chunks long
documents into fixed-size blocks suitable for pretraining.

Tokenizer choice: GPT-2's BPE tokenizer (via Hugging Face AutoTokenizer).
  - Rationale: GPT-2's tokenizer is a well-tested, widely-used BPE
    implementation, vocab_size=50257, and is compatible with most
    GPT-style causal language models. Swap `--tokenizer_name` to use
    WordPiece (e.g. "bert-base-uncased") or any other HF tokenizer.

Output: a single .pt file (PyTorch tensor) containing all token blocks,
plus a small human-readable sample for inspection.

Requires internet (to download the tokenizer files, ~a few MB, cached
after first run) and `transformers` + `torch`.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def chunk_tokens(token_ids: list[int], block_size: int, stride: int | None = None):
    """
    Split a long list of token ids into fixed-size blocks.

    stride=None -> non-overlapping chunks (block_size step).
    stride < block_size -> overlapping chunks (helps the model see context
    that would otherwise be split across a chunk boundary).
    """
    stride = stride or block_size
    chunks = []
    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start:start + block_size]
        # Only drop a chunk for being "too small" if it's a leftover TAIL
        # chunk (i.e. we've already produced at least one full chunk from
        # this document). A short document's *only* chunk must always be
        # kept, otherwise every doc shorter than block_size//4 tokens would
        # silently vanish from the dataset.
        if chunks and len(chunk) < block_size // 4:
            break
        chunks.append(chunk)
        if start + block_size >= len(token_ids):
            break
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="./data/cleaned.jsonl")
    parser.add_argument("--out_path", type=str, default="./data/tokenized_blocks.pt")
    parser.add_argument("--sample_out", type=str, default="./data/sample_dataset.pt")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--block_size", type=int, default=512,
                         help="Max sequence length per training example.")
    parser.add_argument("--stride", type=int, default=None,
                         help="Overlap stride for chunking; defaults to block_size (no overlap).")
    parser.add_argument("--n_sample_blocks", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        # GPT-2 has no pad token by default; reuse EOS as pad for batching later.
        tokenizer.pad_token = tokenizer.eos_token

    # GPT-2's tokenizer warns "Token indices sequence length is longer than
    # the specified maximum sequence length for this model (N > 1024)" for
    # every document over 1024 tokens. That warning assumes you're about to
    # feed the full sequence straight into the GPT-2 *model*. We aren't --
    # we tokenize the full document, then chunk it ourselves into block_size
    # pieces below -- so the 1024 model limit doesn't apply here and the
    # warning is just noise. Raising model_max_length silences it without
    # changing any actual tokenization behavior (encode() never truncated
    # anything either way).
    tokenizer.model_max_length = int(1e12)

    all_blocks = []
    domain_counts = {}

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record["text"]
            domain = record.get("domain", "unknown")

            token_ids = tokenizer.encode(text)
            blocks = chunk_tokens(token_ids, args.block_size, args.stride)
            all_blocks.extend(blocks)
            domain_counts[domain] = domain_counts.get(domain, 0) + len(blocks)

    print(f"[tokenize] tokenizer={args.tokenizer_name} vocab_size={tokenizer.vocab_size}")
    print(f"[tokenize] total blocks produced: {len(all_blocks)}")
    print(f"[tokenize] blocks per domain: {domain_counts}")

    # Pad the final (possibly short) chunk in each block to block_size so we
    # can stack everything into a single rectangular tensor.
    pad_id = tokenizer.pad_token_id
    padded = torch.full((len(all_blocks), args.block_size), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_blocks), args.block_size), dtype=torch.long)
    for i, block in enumerate(all_blocks):
        padded[i, :len(block)] = torch.tensor(block, dtype=torch.long)
        attention_mask[i, :len(block)] = 1

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": padded, "attention_mask": attention_mask}, args.out_path)
    print(f"[tokenize] saved full tokenized dataset -> {args.out_path}  shape={tuple(padded.shape)}")

    # Save a small sample for submission (Section 5.2 of the assignment).
    # IMPORTANT: .clone() is required here. padded[:n] is a VIEW into the
    # full tensor's underlying storage, not a copy -- without .clone(),
    # torch.save() serializes the ENTIRE underlying storage buffer (all
    # 656K+ rows), producing a multi-GB "sample" file even though .shape
    # reports (10, 512). .clone() forces a fresh, small, independent
    # allocation containing only the sliced rows.
    n = min(args.n_sample_blocks, padded.shape[0])
    Path(args.sample_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"input_ids": padded[:n].clone(), "attention_mask": attention_mask[:n].clone()},
        args.sample_out,
    )
    print(f"[tokenize] saved sample ({n} blocks) -> {args.sample_out}")


if __name__ == "__main__":
    main()