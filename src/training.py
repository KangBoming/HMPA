"""Training utilities for peptide bioactivity prediction."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from .models import MLPClassifier

PathLike = Union[str, Path]


class TrainingConfig:
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
        batch_size=1024,
        epochs=30,
        learning_rate=1e-4,
        num_folds=10,
        random_seed=42,
        device="cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.device = device


def evaluate_model(model: MLPClassifier, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    probabilities = []  # type: List[np.ndarray]
    labels = []  # type: List[np.ndarray]
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.append(probs.reshape(-1))
            labels.append(targets.detach().cpu().numpy().reshape(-1))

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float("nan")
    return auroc, auprc


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_path: PathLike,
) -> Tuple[float, float]:
    model = MLPClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    all_labels = [batch_targets for _, batch_targets in train_loader]
    concatenated = torch.cat(all_labels)
    n_pos = int((concatenated == 1).sum().item())
    n_neg = int((concatenated == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=config.device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auprc = -np.inf
    best_metrics = (float("nan"), float("nan"))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(config.epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        auroc, auprc = evaluate_model(model, val_loader, config.device)
        if auprc > best_auprc:
            best_auprc = auprc
            best_metrics = (auroc, auprc)
            torch.save(model.state_dict(), model_path)

    return best_metrics


def train_all_tasks(
    label_df: pd.DataFrame,
    features: torch.Tensor,
    output_dir: PathLike,
    config: TrainingConfig,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_columns = [column for column in label_df.columns if column != "sequence"]
    records = []  # type: List[Dict[str, object]]

    dataset_size = len(label_df)
    if dataset_size != len(features):
        raise ValueError("The number of feature rows must match the number of label rows.")

    for task_name in tqdm(task_columns, desc="Training tasks"):
        labels = torch.tensor(label_df[task_name].values, dtype=torch.float32)
        dataset = TensorDataset(features, labels)
        splitter = StratifiedKFold(
            n_splits=config.num_folds,
            shuffle=True,
            random_state=config.random_seed,
        )

        fold_aurocs = []  # type: List[float]
        fold_auprcs = []  # type: List[float]
        best_fold_auprc = -np.inf
        best_model_filename = ""

        for fold_index, (train_idx, val_idx) in enumerate(
            splitter.split(np.arange(len(dataset)), label_df[task_name].values),
            start=1,
        ):
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=config.batch_size,
                shuffle=False,
            )

            model_filename = f"{task_name}_fold{fold_index}.pt"
            model_path = output_dir / "models" / task_name / model_filename
            auroc, auprc = train_one_fold(train_loader, val_loader, config, model_path)

            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)

            if auprc > best_fold_auprc:
                best_fold_auprc = auprc
                best_model_filename = model_filename

            records.append(
                {
                    "Task": task_name,
                    "Fold": f"fold_{fold_index}",
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "best_model_path": best_model_filename,
                    "best_AUPRC": best_fold_auprc,
                }
            )

        task_mask = [record["Task"] == task_name for record in records]
        auroc_mean = float(np.nanmean(fold_aurocs))
        auroc_std = float(np.nanstd(fold_aurocs))
        auprc_mean = float(np.nanmean(fold_auprcs))
        auprc_std = float(np.nanstd(fold_auprcs))
        pos_ratio = float(label_df[task_name].mean())

        for record, matches_task in zip(records, task_mask):
            if matches_task:
                record["AUROC_mean"] = auroc_mean
                record["AUROC_std"] = auroc_std
                record["AUPRC_mean"] = auprc_mean
                record["AUPRC_std"] = auprc_std
                record["best_model_path"] = best_model_filename
                record["best_AUPRC"] = best_fold_auprc
                record["pos_ratio"] = pos_ratio

    performance_df = pd.DataFrame(records)
    performance_df.to_csv(output_dir / "performance.csv", index=False)
    return performance_df
