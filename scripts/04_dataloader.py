"""
Assignment 1 - Step 4: Custom PyTorch Data Loader
====================================================
Two dataset implementations are provided:

1. TokenBlockDataset (map-style)
   - Loads the whole tokenized .pt file into memory.
   - Simple, fast random-access indexing -- fine when the tokenized
     dataset fits comfortably in RAM.

2. StreamingTokenBlockDataset (iterable-style)
   - Reads the *cleaned* JSONL file line-by-line and tokenizes on the fly,
     never materializing the full dataset in memory.
   - Required for corpora too large to fit in RAM (Section 4.3: "Enable
     iterable streaming for large datasets to avoid memory bottlenecks").

Both plug into a standard torch.utils.data.DataLoader for batching +
shuffling.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 1. Map-style dataset: whole tokenized tensor fits in memory
# ---------------------------------------------------------------------------
class TokenBlockDataset(Dataset):
    """Wraps a pre-tokenized .pt file (produced by 03_tokenization.py)."""

    def __init__(self, tokenized_path: str):
        data = torch.load(tokenized_path)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            # Causal LM labels: same as input_ids (shifted internally by the
            # model/loss function, e.g. GPT-2's forward() does the shifting).
            "labels": self.input_ids[idx].clone(),
        }


# ---------------------------------------------------------------------------
# 2. Iterable-style dataset: streams from disk, tokenizes on the fly
# ---------------------------------------------------------------------------
class StreamingTokenBlockDataset(IterableDataset):
    """
    Streams cleaned documents from a JSONL file, tokenizes them lazily, and
    yields fixed-size blocks -- without ever loading the full dataset into
    memory. Suitable for corpora that are many GB in size.
    """

    def __init__(self, jsonl_path: str, tokenizer_name: str = "gpt2", block_size: int = 512):
        self.jsonl_path = jsonl_path
        self.block_size = block_size
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None  # lazily created per-worker (see __iter__)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def __iter__(self):
        tokenizer = self._get_tokenizer()
        pad_id = tokenizer.pad_token_id

        worker_info = torch.utils.data.get_worker_info()
        # If using multiple DataLoader workers, shard the file across workers
        # so each worker reads a disjoint slice of lines (avoids duplicate work).
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_workers != worker_id:
                    continue
                record = json.loads(line)
                token_ids = tokenizer.encode(record["text"])

                for start in range(0, len(token_ids), self.block_size):
                    chunk = token_ids[start:start + self.block_size]
                    if len(chunk) < self.block_size:
                        # pad the final chunk of this document
                        chunk = chunk + [pad_id] * (self.block_size - len(chunk))
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    attention_mask = (input_ids != pad_id).long()
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": input_ids.clone(),
                    }


# ---------------------------------------------------------------------------
# Collate function (used for both dataset types; handles batching cleanly)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def build_map_style_loader(tokenized_path: str, batch_size: int = 8, shuffle: bool = True, num_workers: int = 2):
    dataset = TokenBlockDataset(tokenized_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,          # random-access shuffling -- only possible for map-style datasets
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def build_streaming_loader(jsonl_path: str, tokenizer_name: str = "gpt2",
                            block_size: int = 512, batch_size: int = 8, num_workers: int = 2):
    dataset = StreamingTokenBlockDataset(jsonl_path, tokenizer_name, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,             # IterableDataset: shuffling must be handled
                                    # inside __iter__ (e.g. shuffle buffer) if needed
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["map", "stream"], default="map")
    parser.add_argument("--tokenized_path", type=str, default="./data/tokenized_blocks.pt")
    parser.add_argument("--jsonl_path", type=str, default="./data/cleaned.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "map":
        loader = build_map_style_loader(args.tokenized_path, batch_size=args.batch_size)
    else:
        loader = build_streaming_loader(args.jsonl_path, batch_size=args.batch_size)

    for i, batch in enumerate(loader):
        print(f"batch {i}: input_ids={tuple(batch['input_ids'].shape)} "
              f"attention_mask={tuple(batch['attention_mask'].shape)}")
        if i >= 2:
            break
