"""Inference utilities for microprotein bioactivity prediction."""

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .models import MLPClassifier

PathLike = Union[str, Path]


def predict_scores(
    dataloader: DataLoader,
    model: MLPClassifier,
    device: str,
) -> torch.Tensor:
    model.to(device)
    model.eval()
    all_scores = []  # type: List[torch.Tensor]
    with torch.inference_mode():
        for (features,) in dataloader:
            features = features.to(device)
            logits = model(features)
            scores = torch.sigmoid(logits).detach().cpu()
            all_scores.append(scores)
    return torch.cat(all_scores)


def run_inference(
    features: torch.Tensor,
    item_ids: List[str],
    performance_csv: PathLike,
    models_root: PathLike,
    output_path: PathLike,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    if len(features) != len(item_ids):
        raise ValueError("The number of ids must match the number of feature rows.")

    performance_df = pd.read_csv(performance_csv)
    best_models = performance_df[["Task", "best_model_path"]].drop_duplicates()
    task_to_model = dict(zip(best_models["Task"], best_models["best_model_path"]))

    dataloader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    result_table = {}  # type: Dict[str, List[float]]

    for task_name, model_filename in tqdm(task_to_model.items(), desc="Running inference"):
        model_path = Path(models_root) / task_name / model_filename
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        scores = predict_scores(dataloader, model, device)
        result_table[task_name] = scores.numpy().tolist()

    result_df = pd.DataFrame(result_table, index=item_ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".pkl":
        result_df.to_pickle(output_path)
    else:
        result_df.to_csv(output_path)
    return result_df
