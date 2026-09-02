"""
PyTorch neural network architectures and baseline models for trading.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path


def get_device() -> torch.device:
    """Auto-detect best available compute accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TradingDataset(Dataset):
    """PyTorch Dataset for financial feature matrices and target vectors."""

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray
    ):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TradingDNN(nn.Module):
    """Deep Neural Network for directional market classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass outputting probabilities $p \in (0, 1)$."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def train_trading_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    save_path: str | Path | None = None,
    verbose: bool = True
) -> dict[str, list[float]]:
    """Train PyTorch trading model with binary cross entropy with logits."""
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = None, None
        if val_loader is not None:
            model.eval()
            v_total_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    v_total_loss += loss.item() * len(batch_y)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    v_correct += (preds == batch_y).sum().item()
                    v_total += len(batch_y)

            val_loss = v_total_loss / v_total
            val_acc = v_correct / v_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss and save_path is not None:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            val_str = ""
            if val_loss is not None:
                val_str = (
                    f" | Val Loss: {val_loss:.4f} - "
                    f"Val Acc: {val_acc:.2%}"
                )
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.2%}{val_str}"
            )

    return history
