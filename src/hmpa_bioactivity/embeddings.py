"""ESM-2 embedding extraction utilities."""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm


def read_fasta_records(fasta_path):
    """Read FASTA records as (id, sequence) tuples."""
    records = []
    current_id = None
    current_sequence = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_sequence)))
                current_id = line[1:].split()[0]
                current_sequence = []
            else:
                current_sequence.append(line)
    if current_id is not None:
        records.append((current_id, "".join(current_sequence)))
    return records


def mean_pool_hidden_states(hidden_states, attention_mask):
    """Mean pool token embeddings while excluding padding and special tokens."""
    mask = attention_mask.bool()
    pooled_rows = []
    for row_index in range(hidden_states.size(0)):
        token_embeddings = hidden_states[row_index][mask[row_index]]
        if token_embeddings.size(0) > 2:
            token_embeddings = token_embeddings[1:-1]
        pooled_rows.append(token_embeddings.mean(dim=0, keepdim=True))
    return torch.cat(pooled_rows, dim=0)


def batch_iterator(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_esm2_embeddings(
    records,
    tokenizer,
    model,
    device,
    batch_size,
    pooling="mean",
    max_length=1024,
):
    """Extract ESM-2 embeddings for FASTA records."""
    ids = []
    embedding_chunks = []

    model.eval()
    model.to(device)

    for batch in tqdm(list(batch_iterator(records, batch_size)), desc="Extracting embeddings"):
        batch_ids = [item[0] for item in batch]
        batch_sequences = [item[1] for item in batch]
        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state

        if pooling == "mean":
            pooled = mean_pool_hidden_states(hidden_states, encoded["attention_mask"])
        elif pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError("pooling must be either 'mean' or 'cls'")

        ids.extend(batch_ids)
        embedding_chunks.append(pooled.cpu())

    embeddings = torch.cat(embedding_chunks, dim=0)
    return ids, embeddings

