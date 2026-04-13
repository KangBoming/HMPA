"""Model definitions used in the HMPA bioactivity repository."""

import torch
from torch import nn
from typing import List


class MLPClassifier(nn.Module):
    """Simple multilayer perceptron used for task-wise binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []  # type: List[nn.Module]
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)
