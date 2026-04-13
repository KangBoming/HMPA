"""Input and output helpers for training and inference."""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch


PathLike = Union[str, Path]


def load_label_table(label_csv: PathLike) -> pd.DataFrame:
    df = pd.read_csv(label_csv)
    if "sequence" not in df.columns:
        raise ValueError("The label CSV must contain a 'sequence' column.")
    return df


def load_feature_tensor(feature_path: PathLike) -> torch.Tensor:
    features = torch.load(feature_path, map_location="cpu")
    if not isinstance(features, torch.Tensor):
        raise TypeError("Expected feature_path to contain a torch.Tensor.")
    return features.float()


def load_embedding_matrix(
    embedding_path: PathLike,
    ids: Optional[Iterable[str]] = None,
) -> Tuple[List[str], torch.Tensor]:
    embeddings = torch.load(embedding_path, map_location="cpu")

    if isinstance(embeddings, torch.Tensor):
        if ids is None:
            generated_ids = [f"sample_{idx}" for idx in range(len(embeddings))]
            return generated_ids, embeddings.float()
        id_list = list(ids)
        if len(id_list) != len(embeddings):
            raise ValueError("The number of ids does not match the number of embeddings.")
        return id_list, embeddings.float()

    if isinstance(embeddings, dict):
        if ids is None:
            raise ValueError("When embedding_path stores a dict, ids must be supplied.")
        id_list = list(ids)
        feature_list = []  # type: List[torch.Tensor]
        retained_ids = []  # type: List[str]
        for item_id in id_list:
            if item_id in embeddings:
                tensor = embeddings[item_id]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Embedding for {item_id} is not a torch.Tensor.")
                feature_list.append(tensor.detach().cpu().float().unsqueeze(0))
                retained_ids.append(item_id)
        if not feature_list:
            raise ValueError("No matching ids were found in the embedding dictionary.")
        return retained_ids, torch.cat(feature_list, dim=0)

    raise TypeError("embedding_path must contain either a torch.Tensor or a dict of tensors.")


def read_fasta_ids(fasta_path: PathLike) -> List[str]:
    ids = []  # type: List[str]
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line.strip()[1:])
    return ids


def read_id_list(id_path: PathLike) -> List[str]:
    with open(id_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]
